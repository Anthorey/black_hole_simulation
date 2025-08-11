#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlgpu3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "config.h"
#include "shader.hpp"

static constexpr float kPan = 0.002f;
static constexpr float kZoom = 25.0e9f;
static constexpr float kFov = glm::radians<float>(60.0f);

struct UniformBuffer
{
    glm::vec3 CameraPosition;
    float TanHalfFov;
    glm::vec3 CameraRight;
    float Aspect;
    glm::vec3 CameraUp;
    float Padding0;
    glm::vec3 CameraForward;
    float Padding1;
};

static SDL_Window* window;
static SDL_GPUDevice* device;
static SDL_GPUComputePipeline* geodesicPipeline;
static SDL_GPUTexture* colorTexture;
static float pitch;
static float yaw;
static float distance{1.0e11f};
static UniformBuffer uniformBuffer;

static bool Init()
{
    SDL_SetAppMetadata("Black Hole Simulation", nullptr, nullptr);
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return false;
    }
    window = SDL_CreateWindow("Black Hole Simulation", 960, 720, SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return false;
    }
#if defined(SDL_PLATFORM_WIN32)
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, nullptr);
#elif defined(SDL_PLATFORM_APPLE)
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, true, nullptr);
#else
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
#endif
    if (!device)
    {
        SDL_Log("Failed to create device: %s", SDL_GetError());
        return false;
    }
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("Failed to create swapchain: %s", SDL_GetError());
        return false;
    }
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplSDL3_InitForSDLGPU(window);
        ImGui_ImplSDLGPU3_InitInfo info{};
        info.Device = device;
        info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
        ImGui_ImplSDLGPU3_Init(&info);
    }
    geodesicPipeline = LoadComputePipeline(device, "geodesic.comp");
    if (!geodesicPipeline)
    {
        SDL_Log("Failed to create geodesic pipeline");
        return false;
    }
    {
        SDL_GPUTextureCreateInfo info{};
        info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        info.usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER;
        info.type = SDL_GPU_TEXTURETYPE_2D;
        info.width = WIDTH;
        info.height = HEIGHT;
        info.layer_count_or_depth = 1;
        info.num_levels = 1;
        colorTexture = SDL_CreateGPUTexture(device, &info);
        if (!colorTexture)
        {
            SDL_Log("Failed to create color texture: %s", SDL_GetError());
            return false;
        }
    }
    return true;
}

static void Draw()
{
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(device);
    if (!commandBuffer)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return;
    }
    SDL_GPUTexture* swapchainTexture;
    uint32_t width;
    uint32_t height;
    if (!SDL_AcquireGPUSwapchainTexture(commandBuffer, window, &swapchainTexture, &width, &height))
    {
        SDL_Log("Failed to acquire swapchain texture: %s", SDL_GetError());
        SDL_CancelGPUCommandBuffer(commandBuffer);
        return;
    }
    if (!swapchainTexture || !width || !height)
    {
        /* NOTE: not an error */
        SDL_SubmitGPUCommandBuffer(commandBuffer);
        return;
    }
    {
        uniformBuffer.CameraUp = glm::vec3{0.0f, 1.0f, 0.0f};
        uniformBuffer.TanHalfFov = std::tan(kFov * 0.5f);
        uniformBuffer.Aspect = float(WIDTH) / HEIGHT;
        uniformBuffer.CameraForward.x = std::cos(pitch) * std::cos(yaw);
        uniformBuffer.CameraForward.y = std::sin(pitch);
        uniformBuffer.CameraForward.z = std::cos(pitch) * std::sin(yaw);
        uniformBuffer.CameraForward = glm::normalize(uniformBuffer.CameraForward);
        uniformBuffer.CameraPosition = -uniformBuffer.CameraForward * distance;
        uniformBuffer.CameraRight = glm::cross(uniformBuffer.CameraForward, uniformBuffer.CameraUp);
        uniformBuffer.CameraRight = glm::normalize(uniformBuffer.CameraRight);
    }
    {
        SDL_GPUStorageTextureReadWriteBinding readWriteTexture{};
        readWriteTexture.texture = colorTexture;
        SDL_GPUComputePass* computePass = SDL_BeginGPUComputePass(commandBuffer, &readWriteTexture, 1, nullptr, 0);
        if (!computePass)
        {
            SDL_Log("Failed to begin compute pass: %s", SDL_GetError());
            SDL_SubmitGPUCommandBuffer(commandBuffer);
            return;
        }
        int groupsX = (WIDTH + THREADS - 1) / THREADS;
        int groupsY = (HEIGHT + THREADS - 1) / THREADS;
        SDL_BindGPUComputePipeline(computePass, geodesicPipeline);
        SDL_PushGPUComputeUniformData(commandBuffer, 0, &uniformBuffer, sizeof(uniformBuffer));
        SDL_DispatchGPUCompute(computePass, groupsX, groupsY, 1);
        SDL_EndGPUComputePass(computePass);
    }
    {
        uint32_t letterboxW;
        uint32_t letterboxH;
        uint32_t letterboxX;
        uint32_t letterboxY;
        if ((static_cast<float>(WIDTH) / HEIGHT) > (static_cast<float>(width) / height))
        {
            letterboxW = width;
            letterboxH = HEIGHT * static_cast<float>(width) / WIDTH;
            letterboxX = 0.0f;
            letterboxY = (height - letterboxH) / 2.0f;
        }
        else
        {
            letterboxH = height;
            letterboxW = WIDTH * static_cast<float>(height) / HEIGHT;
            letterboxX = (width - letterboxW) / 2.0f;
            letterboxY = 0.0f;
        }
        SDL_FColor clearColor = {0.02f, 0.02f, 0.02f, 1.0f};
        SDL_GPUBlitInfo info{};
        info.load_op = SDL_GPU_LOADOP_CLEAR;
        info.clear_color = clearColor;
        info.source.texture = colorTexture;
        info.source.w = WIDTH;
        info.source.h = HEIGHT;
        info.destination.texture = swapchainTexture;
        info.destination.x = letterboxX;
        info.destination.y = letterboxY;
        info.destination.w = letterboxW;
        info.destination.h = letterboxH;
        info.filter = SDL_GPU_FILTER_NEAREST;
        SDL_BlitGPUTexture(commandBuffer, &info);
    }
    SDL_SubmitGPUCommandBuffer(commandBuffer);
}

int main(int argc, char** argv)
{
    if (!Init())
    {
        return 1;
    }
    uint64_t time2 = SDL_GetTicks();
    uint64_t time1 = time2;
    bool running = true;
    while (running)
    {
        time2 = SDL_GetTicks();
        float delta = time2 - time1;
        time1 = time2;
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            switch (event.type)
            {
            case SDL_EVENT_MOUSE_WHEEL:
                distance = std::max(1.0f, distance - event.wheel.y * kZoom * delta);
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (event.motion.state & SDL_BUTTON_LMASK)
                {
                    static constexpr float kClamp = glm::pi<float>() / 2.0f - 0.01f;
                    yaw += event.motion.xrel * kPan * delta;
                    pitch = std::clamp(pitch - event.motion.yrel * kPan * delta, -kClamp, kClamp);
                }
                break;
            case SDL_EVENT_QUIT:
                running = false;
                break;
            }
        }
        Draw();
    }
    SDL_HideWindow(window);
    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_ReleaseGPUTexture(device, colorTexture);
    SDL_ReleaseGPUComputePipeline(device, geodesicPipeline);
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}