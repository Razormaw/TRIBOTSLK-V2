#include "ScriptMgr.h"
#include "Player.h"
#include "Channel.h"
#include "ChannelMgr.h"
#include "Chat.h"
#include "ChatPackets.h"
#include "ObjectMgr.h"
#include "World.h"
#include "Config.h"
#include "NpcBotLLMInterface.h"
#include "botmgr.h"
#include "bot_ai.h"
#include "botdatamgr.h"
#include "botlog.h"
#include <cctype>
#include <thread>
#include <mutex>
#include <queue>
#include <random>
#include <algorithm>
#include "DatabaseEnv.h"
#include "Opcodes.h"
#include "Map.h"

struct PendingBotChat { ObjectGuid botGuid; std::string reply; std::string channelName; };
static std::queue<PendingBotChat> g_pendingBotChats;
static std::mutex g_pendingBotChatsMutex;

struct BotConversation { ObjectGuid botA; ObjectGuid botB; std::string seedPhrase; uint8 turnsLeft; bool nextIsBotA; };
static std::queue<BotConversation> g_activeConversations;
static std::mutex g_conversationMutex;
static uint32 g_botToBotTimer = 0;

static std::vector<std::string> g_phrasesStart;
static std::vector<std::string> g_phrasesResponse;
static bool g_phrasesLoaded = false;

static void LoadPhrasesFromDB()
{
    g_phrasesStart.clear(); g_phrasesResponse.clear();
    QueryResult result = WorldDatabase.Query("SELECT category, text FROM npcbot_llm_phrases");
    if (result) {
        do {
            Field* fields = result->Fetch();
            std::string category = fields[0].GetString();
            std::string text = fields[1].GetString();
            if (category == "inicio") g_phrasesStart.push_back(text);
            else if (category == "respuesta") g_phrasesResponse.push_back(text);
        } while (result->NextRow());
    }
    g_phrasesLoaded = true;
    BOT_LOG_INFO("npcbots", "LLM: frases cargadas: {} inicio, {} respuesta.", (uint32)g_phrasesStart.size(), (uint32)g_phrasesResponse.size());
}

struct ChainTurn { ObjectGuid botGuid; std::string seedText; std::string seedAuthor; std::string playerName; std::vector<ObjectGuid> nextBots; uint8 turnsLeft; std::string channelName; };
static std::queue<ChainTurn> g_playerChains;
static std::mutex g_chainMutex;

static Creature* FindBotByGuid(ObjectGuid guid)
{
    NpcBotRegistry const& registry = BotDataMgr::GetExistingNPCBots();
    for (NpcBotRegistry::const_iterator itr = registry.begin(); itr != registry.end(); ++itr)
        if (*itr && (*itr)->GetGUID() == guid) return const_cast<Creature*>(*itr);
    return nullptr;
}

static void SendBotChannelMessage(Creature* bot, const std::string& channelName, const std::string& msg)
{
    if (!bot || !bot->IsInWorld() || !bot->IsAlive()) return;

    ObjectGuid vGuid = ObjectGuid::Create<HighGuid::Player>(uint32(0x00F00000 + bot->GetEntry()));
    std::string const& botName = bot->GetName();

    // 1. SMSG_NAME_QUERY_RESPONSE
    WorldPacket nameData(SMSG_NAME_QUERY_RESPONSE, botName.size() + 40);
    nameData.appendPackGUID(vGuid.GetRawValue());
    nameData << uint8(0);
    nameData << botName;
    nameData << std::string("");
    nameData << uint8(bot->GetByteValue(UNIT_FIELD_BYTES_0, 0));
    nameData << uint8(bot->GetByteValue(UNIT_FIELD_BYTES_0, 2));
    nameData << uint8(bot->GetByteValue(UNIT_FIELD_BYTES_0, 1));
    nameData << uint8(0);
    sWorld->SendGlobalMessage(&nameData);

    // 2. SMSG_MESSAGECHAT usando la API OFICIAL (Inmune a crashes de Trickerer)
    WorldPackets::Chat::Chat chatPacket;
    chatPacket.Initialize(CHAT_MSG_CHANNEL, LANG_UNIVERSAL, nullptr, nullptr, msg, 0, channelName, DEFAULT_LOCALE);
    chatPacket.SenderGUID = vGuid;
    chatPacket.TargetGUID = vGuid;
    sWorld->SendGlobalMessage(chatPacket.Write());

    BOT_LOG_INFO("npcbots", "LLM: [{}] {}: {}", channelName, bot->GetName(), msg);
}

static std::string BuildLLMRequestBody(const std::string& prompt)
{
    std::string model = sConfigMgr->GetStringDefault("NpcBot.LLM.Model", "qwen/qwen3.8-27b");
    int maxTokens = sConfigMgr->GetIntDefault("NpcBot.LLM.MaxTokens", 120);
    std::string body = "{";
    body += "\"model\":\"" + NpcBotLLMInterface::SanitizeForJson(model) + "\",";
    body += "\"max_tokens\":" + std::to_string(maxTokens) + ",";
    body += "\"messages\":[";
    body += "{\"role\":\"system\",\"content\":\"Eres un NPC de WoW 3.3.5. Responde breve, sin comillas.\"},";
    body += "{\"role\":\"user\",\"content\":\"" + NpcBotLLMInterface::SanitizeForJson(prompt) + "\"}";
    body += "]}";
    return body;
}

static std::string ExtractOpenAIResponse(const std::string& json)
{
    size_t choicesStart = json.find("\"choices\"");
    if (choicesStart == std::string::npos) return "";
    size_t contentStart = json.find("\"content\"", choicesStart);
    if (contentStart == std::string::npos) return "";
    contentStart = json.find(":", contentStart);
    if (contentStart == std::string::npos) return "";
    size_t valueStart = json.find("\"", contentStart);
    if (valueStart == std::string::npos) return "";
    valueStart++;
    std::string out;
    for (size_t i = valueStart; i < json.size(); ++i)
    {
        char c = json[i];
        if (c == '\\' && i + 1 < json.size())
        {
            char n = json[++i];
            if (n == 'n') out += '\n';
            else if (n == 't') out += '\t';
            else if (n == 'r') out += '\r';
            else if (n == '"') out += '"';
            else if (n == '\\') out += '\\';
            else out += n;
        }
        else if (c == '"') break;
        else out += c;
    }
    return out;
}

class npcbot_llm_chat_player_script : public PlayerScript
{
public:
    npcbot_llm_chat_player_script() : PlayerScript("npcbot_llm_chat_player_script") {}

    void OnChat(Player* player, uint32 type, uint32 /*lang*/, std::string& msg, Channel* channel) override
    {
        if (type != CHAT_MSG_CHANNEL || !channel) return;
        if (!sConfigMgr->GetBoolDefault("NpcBot.LLM.Enable", false)) return;

        std::string channelName = channel->GetName();
        std::string lower = channelName;
        for (size_t i = 0; i < lower.size(); ++i) lower[i] = (char)tolower((unsigned char)lower[i]);

        std::string channelsConf = sConfigMgr->GetStringDefault("NpcBot.LLM.Channels", "world,mundo,general");
        bool allowed = false;
        size_t pos = 0;
        while (pos <= channelsConf.size())
        {
            size_t comma = channelsConf.find(',', pos);
            std::string token = channelsConf.substr(pos, (comma == std::string::npos ? channelsConf.size() : comma) - pos);
            size_t b = token.find_first_not_of(" \t");
            size_t e = token.find_last_not_of(" \t");
            if (b != std::string::npos)
            {
                token = token.substr(b, e - b + 1);
                for (size_t i = 0; i < token.size(); ++i) token[i] = (char)tolower((unsigned char)token[i]);
                if (token == lower) { allowed = true; break; }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (!allowed) return;

        std::vector<Creature*> availableBots;
        NpcBotRegistry const& registry = BotDataMgr::GetExistingNPCBots();
        for (NpcBotRegistry::const_iterator itr = registry.begin(); itr != registry.end(); ++itr)
        {
            Creature* bot = const_cast<Creature*>(*itr);
            if (bot && bot->IsInWorld() && bot->IsAlive() && !bot->IsInCombat()) availableBots.push_back(bot);
        }
        if (availableBots.empty()) return;

        std::random_device rd; std::mt19937 gen(rd());
        std::shuffle(availableBots.begin(), availableBots.end(), gen);

        uint8 minT = uint8(sConfigMgr->GetIntDefault("NpcBot.LLM.PlayerChain.MinTurns", 2));
        uint8 maxT = uint8(sConfigMgr->GetIntDefault("NpcBot.LLM.PlayerChain.MaxTurns", 3));
        if (maxT < minT) maxT = minT;
        uint8 turns = uint8(urand(minT, maxT));
        if (!sConfigMgr->GetBoolDefault("NpcBot.LLM.PlayerChain.Enable", true)) turns = 1;
        if (turns > availableBots.size()) turns = uint8(availableBots.size());

        ChainTurn turn;
        turn.botGuid = availableBots[0]->GetGUID();
        turn.seedText = msg;
        turn.seedAuthor = player->GetName();
        turn.playerName = player->GetName();
        turn.turnsLeft = uint8(turns - 1);
        turn.channelName = channelName;
        for (uint8 i = 1; i < turns; ++i) turn.nextBots.push_back(availableBots[i]->GetGUID());

        { std::lock_guard<std::mutex> lock(g_chainMutex); g_playerChains.push(turn); }
    }
};

class npcbot_llm_chat_world_script : public WorldScript
{
public:
    npcbot_llm_chat_world_script() : WorldScript("npcbot_llm_chat_world_script") {}

    void OnUpdate(uint32 diff) override
    {
        std::queue<PendingBotChat> localQueue;
        { std::lock_guard<std::mutex> lock(g_pendingBotChatsMutex); if (!g_pendingBotChats.empty()) std::swap(localQueue, g_pendingBotChats); }
        while (!localQueue.empty())
        {
            PendingBotChat task = localQueue.front(); localQueue.pop();
            Creature* bot = FindBotByGuid(task.botGuid);
            if (bot && bot->IsInWorld()) SendBotChannelMessage(bot, task.channelName, task.reply);
        }

        {
            bool hasChain;
            { std::lock_guard<std::mutex> lock(g_chainMutex); hasChain = !g_playerChains.empty(); }
            if (hasChain)
            {
                ChainTurn turn;
                { std::lock_guard<std::mutex> lock(g_chainMutex); turn = g_playerChains.front(); g_playerChains.pop(); }
                Creature* speaker = FindBotByGuid(turn.botGuid);
                if (speaker && speaker->IsInWorld() && speaker->IsAlive())
                {
                    struct ChainData { ObjectGuid botGuid; std::string botName, seedText, seedAuthor, playerName; std::vector<ObjectGuid> nextBots; uint8 turnsLeft; };
                    ChainData* cd = new ChainData();
                    cd->botGuid = speaker->GetGUID(); cd->botName = speaker->GetName(); cd->seedText = turn.seedText; cd->seedAuthor = turn.seedAuthor;
                    cd->playerName = turn.playerName; cd->nextBots = turn.nextBots; cd->turnsLeft = turn.turnsLeft;

                    std::thread([channelName = turn.channelName](ChainData* d)
                    {
                        try
                        {
                            std::string prompt = "Eres un NPC llamado '" + d->botName + "' en WoW 3.3.5a. ";
                            prompt += "Un jugador llamado '" + d->playerName + "' habla en el canal '" + channelName + "'. ";
                            prompt += "El ultimo mensaje fue de '" + d->seedAuthor + "': \"" + d->seedText + "\". ";
                            prompt += "Responde breve (max. 100 caracteres), en personaje. Sin comillas ni markdown.";
                            
                            std::vector<std::string> debugLines;
                            int timeout = sConfigMgr->GetIntDefault("NpcBot.LLM.TimeoutSeconds", 15);
                            int maxGen = sConfigMgr->GetIntDefault("NpcBot.LLM.MaxGenerations", 3);
                            std::string body = BuildLLMRequestBody(prompt);
                            std::string response = NpcBotLLMInterface::Generate(body, timeout, maxGen, debugLines);
                            std::string reply;
                            if (!response.empty() && response.find("\"error\"") == std::string::npos) reply = ExtractOpenAIResponse(response);
                            if (reply.empty() && !g_phrasesResponse.empty()) reply = g_phrasesResponse[urand(0, (int)g_phrasesResponse.size() - 1)];

                            if (!reply.empty())
                            {
                                { std::lock_guard<std::mutex> lock(g_pendingBotChatsMutex); g_pendingBotChats.push({d->botGuid, reply, channelName}); }
                                if (d->turnsLeft > 0 && !d->nextBots.empty())
                                {
                                    std::lock_guard<std::mutex> lock(g_chainMutex);
                                    ChainTurn next;
                                    next.botGuid = d->nextBots.front(); next.seedText = reply; next.seedAuthor = d->botName;
                                    next.playerName = d->playerName; next.channelName = channelName; next.turnsLeft = uint8(d->turnsLeft - 1);
                                    for (size_t i = 1; i < d->nextBots.size(); ++i) next.nextBots.push_back(d->nextBots[i]);
                                    g_playerChains.push(next);
                                }
                            }
                            delete d;
                        } catch (...) { delete d; }
                    }, cd).detach();
                }
            }
        }

        if (sConfigMgr->GetBoolDefault("NpcBot.LLM.BotToBot.Enable", false))
        {
            std::string botChannel = sConfigMgr->GetStringDefault("NpcBot.LLM.BotToBot.Channel", "world");
            if (!g_phrasesLoaded) LoadPhrasesFromDB();

            bool playerQueueEmpty, chainEmpty;
            { std::lock_guard<std::mutex> lock(g_pendingBotChatsMutex); playerQueueEmpty = g_pendingBotChats.empty(); }
            { std::lock_guard<std::mutex> lock(g_chainMutex); chainEmpty = g_playerChains.empty(); }

            g_botToBotTimer += diff;
            uint32 cooldownMs = uint32(sConfigMgr->GetIntDefault("NpcBot.LLM.BotToBot.CooldownSec", 150)) * 1000u;

            if (playerQueueEmpty && chainEmpty && g_botToBotTimer >= cooldownMs)
            {
                g_botToBotTimer = 0;
                uint32 chance = uint32(sConfigMgr->GetIntDefault("NpcBot.LLM.BotToBot.Chance", 25));
                bool convActive;
                { std::lock_guard<std::mutex> lock(g_conversationMutex); convActive = !g_activeConversations.empty(); }

                if (!convActive && !g_phrasesStart.empty() && urand(1, 100) <= chance)
                {
                    std::vector<Creature*> bots;
                    NpcBotRegistry const& registry = BotDataMgr::GetExistingNPCBots();
                    for (NpcBotRegistry::const_iterator itr = registry.begin(); itr != registry.end(); ++itr)
                        if (auto* bot = const_cast<Creature*>(*itr); bot && bot->IsInWorld() && bot->IsAlive() && !bot->IsInCombat()) bots.push_back(bot);

                    if (bots.size() >= 2)
                    {
                        std::random_device rd; std::mt19937 gen(rd()); std::shuffle(bots.begin(), bots.end(), gen);
                        Creature* botA = bots[0]; Creature* botB = bots[1];
                        std::string seed = g_phrasesStart[urand(0, (int)g_phrasesStart.size() - 1)];
                        SendBotChannelMessage(botA, botChannel, seed);

                        uint8 minT = uint8(sConfigMgr->GetIntDefault("NpcBot.LLM.BotToBot.MinTurns", 2));
                        uint8 maxT = uint8(sConfigMgr->GetIntDefault("NpcBot.LLM.BotToBot.MaxTurns", 3));
                        if (maxT < minT) maxT = minT;

                        BotConversation conv;
                        conv.botA = botA->GetGUID(); conv.botB = botB->GetGUID(); conv.seedPhrase = seed;
                        conv.turnsLeft = uint8(urand(minT, maxT)); conv.nextIsBotA = false;
                        { std::lock_guard<std::mutex> lock(g_conversationMutex); g_activeConversations.push(conv); }
                        BOT_LOG_INFO("npcbots", "LLM: charla iniciada: '{}' y '{}'.", botA->GetName(), botB->GetName());
                    }
                }
            }

            bool convActiveNow;
            { std::lock_guard<std::mutex> lock(g_conversationMutex); convActiveNow = !g_activeConversations.empty(); }
            if (playerQueueEmpty && chainEmpty && convActiveNow)
            {
                BotConversation conv; conv.turnsLeft = 0;
                { std::lock_guard<std::mutex> lock(g_conversationMutex); if (!g_activeConversations.empty()) { conv = g_activeConversations.front(); g_activeConversations.pop(); } }
                
                if (conv.turnsLeft > 0)
                {
                    Creature* speaker = FindBotByGuid(conv.nextIsBotA ? conv.botA : conv.botB);
                    Creature* other = FindBotByGuid(conv.nextIsBotA ? conv.botB : conv.botA);
                    if (speaker && other && speaker->IsInWorld() && speaker->IsAlive() && other->IsInWorld() && other->IsAlive())
                    {
                        struct ConvData { ObjectGuid speakerGuid, botA, botB; std::string speakerName, otherName, seed; bool nextIsBotA; uint8 turnsLeft; };
                        ConvData* cd = new ConvData();
                        cd->speakerGuid = speaker->GetGUID(); cd->speakerName = speaker->GetName(); cd->otherName = other->GetName();
                        cd->seed = conv.seedPhrase; cd->botA = conv.botA; cd->botB = conv.botB; cd->nextIsBotA = conv.nextIsBotA; cd->turnsLeft = conv.turnsLeft;

                        std::thread([botChannel](ConvData* d)
                        {
                            try
                            {
                                std::string prompt = "Eres un NPC llamado '" + d->speakerName + "' en WoW 3.3.5a. ";
                                prompt += "Otro NPC llamado '" + d->otherName + "' dijo en el canal '" + botChannel + "': \"" + d->seed + "\". ";
                                prompt += "Responde breve (max. 100 caracteres), en personaje. Sin comillas ni markdown.";
                                
                                std::vector<std::string> debugLines;
                                int timeout = sConfigMgr->GetIntDefault("NpcBot.LLM.TimeoutSeconds", 15);
                                int maxGen = sConfigMgr->GetIntDefault("NpcBot.LLM.MaxGenerations", 3);
                                std::string body = BuildLLMRequestBody(prompt);
                                std::string response = NpcBotLLMInterface::Generate(body, timeout, maxGen, debugLines);
                                std::string reply;
                                if (!response.empty() && response.find("\"error\"") == std::string::npos) reply = ExtractOpenAIResponse(response);
                                if (reply.empty() && !g_phrasesResponse.empty()) reply = g_phrasesResponse[urand(0, (int)g_phrasesResponse.size() - 1)];

                                if (!reply.empty())
                                {
                                    { std::lock_guard<std::mutex> lock(g_pendingBotChatsMutex); g_pendingBotChats.push({d->speakerGuid, reply, botChannel}); }
                                    if (d->turnsLeft > 1)
                                    {
                                        std::lock_guard<std::mutex> lock(g_conversationMutex);
                                        BotConversation next;
                                        next.botA = d->botA; next.botB = d->botB; next.seedPhrase = reply;
                                        next.turnsLeft = uint8(d->turnsLeft - 1); next.nextIsBotA = !d->nextIsBotA;
                                        g_activeConversations.push(next);
                                    }
                                }
                                delete d;
                            } catch (...) { delete d; }
                        }, cd).detach();
                    }
                }
            }
        }
    }
};

TC_GAME_API void AddSC_npcbot_llm_chat()
{
    new npcbot_llm_chat_player_script();
    new npcbot_llm_chat_world_script();
    BOT_LOG_INFO("npcbots", "LLM: scripts registrados.");
}
