#ifndef _NPCBOT_LLM_INTERFACE_H
#define _NPCBOT_LLM_INTERFACE_H

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

struct ParsedUrl
{
    bool https = false;
    std::string hostname;
    uint16_t port = 80;
    std::string path = "/";
};

class NpcBotLLMInterface
{
public:
    NpcBotLLMInterface() = default;
    ~NpcBotLLMInterface() = default;

    static std::string SanitizeForJson(const std::string& input);
    static std::string Generate(const std::string& prompt, int timeOutSeconds, int maxGenerations, std::vector<std::string>& debugLines);
    static std::vector<std::string> ParseResponse(const std::string& response, const std::string& startPattern, const std::string& endPattern, const std::string& deletePattern, const std::string& splitPattern, std::vector<std::string>& debugLines);
    static void LimitContext(std::string& context, uint32_t currentLength);
    static ParsedUrl ParseUrl(const std::string& url);

private:
    static std::atomic<int> generationCount;
};

#endif
