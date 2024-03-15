/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <iostream>
#include <fstream>

#include <Windows.h>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"

#define CURR_DIRECTORY "C:/Users/raffa/Documents/GitHub/ray_tracing/"


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  ImGuiH::CameraWidget();
  if(ImGui::CollapsingHeader("Light"))
  {
    ImGui::RadioButton("Point", &helloVk.m_pcRaster.lightType, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Infinite", &helloVk.m_pcRaster.lightType, 1);

    ImGui::SliderFloat3("Position", &helloVk.m_pcRaster.lightPosition.x, -100.f, 100.f);
    ImGui::SliderFloat("Intensity", &helloVk.m_pcRaster.lightIntensity, 0.f, 150.f);
  }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1920;
static int const SAMPLE_HEIGHT = 1080;


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  /* std::ofstream  out;
  std::string         filename;
  SYSTEM_POWER_STATUS systemStatus;
  GetSystemPowerStatus(&systemStatus);

  if(systemStatus.ACLineStatus == 255)
  {
    filename = "data/avg_framerate_desktop.txt";
    filename = CURR_DIRECTORY + filename;
    out.open(filename);
  }
  else if(systemStatus.ACLineStatus == 1)
  {
    filename = "data/avg_framerate_ac.txt";
    filename = CURR_DIRECTORY + filename;
    out.open(filename);
  }
  else
  {
    filename = "data/avg_framerate_battery.txt";
    filename = CURR_DIRECTORY + filename; 
    out.open(filename);
  }*/

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);


  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat(nvmath::vec3f(0.0f, 0.0f, 0.0f), nvmath::vec3f(10, 0, 0), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(15.0f, 0.0f, 10.0f), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(-10.0f, 0.0f, 13.0f), nvmath::vec3f(0, 0, -10), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(-11.0f, 3.0f, 1.0f), nvmath::vec3f(10, 5, -1), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(-17.0f, 3.0f, 13.0f), nvmath::vec3f(-11, 3, 1), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(-9.0f, 6.0f, 20.0f), nvmath::vec3f(-20, 3, -10), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(23.0f, 5.0f, -16.0f), nvmath::vec3f(23, 5, -30), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(23.0f, 6.0f, -47.0f), nvmath::vec3f(23, 6, -30), nvmath::vec3f(0, 1, 0));
  // CameraManip.setLookat(nvmath::vec3f(35.0f, 9.0f, -41.0f), nvmath::vec3f(12, 9, -30), nvmath::vec3f(0, 1, 0));
  CameraManip.setSpeed(1500.f);

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // #VKRay: activate ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);   // extension required to build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);    // extension required to use ckCmdTraceRaysKHR
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);   //extension required by ray tracing pipeline (defines the infrastructure and usage patterns for deferrable commands, but does not specify any commands as deferrable)


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  HelloVulkan helloVk;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0

  // Subject
  helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths, true));

  // First Scene - simple
  helloVk.loadModel(nvh::findFile("media/scenes/Medieval_building.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(0, -1, 10)));
  helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(0, -1, -10)));
  helloVk.loadModel(nvh::findFile("media/scenes/sphere.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(10, 0, 0)) * nvmath::scale_mat4(nvmath::vec3f(2.f, 2.f, 2.f)));
  helloVk.loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(-10, 0, 0)) * nvmath::scale_mat4(nvmath::vec3f(2.f, 2.f, 2.f)));
  helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(0, -1, 0)));

  // Second scene - city
  /* helloVk.loadModel(nvh::findFile("media/scenes/Center_City_SciFi.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(0, 0, 0)) * nvmath::scale_mat4(nvmath::vec3f(0.1f, 0.1f, 0.1f)));*/

  // Third scene - house
  // helloVk.loadModel(nvh::findFile("media/scenes/House01.obj", defaultSearchPaths, true));

  helloVk.createOffscreenRender();
  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline();
  helloVk.createUniformBuffer();
  helloVk.createObjDescriptionBuffer();
  helloVk.updateDescriptorSet();

  // #VKRay
  helloVk.initRayTracing();
  helloVk.createBottomLevelAS();
  helloVk.createTopLevelAS();
  helloVk.createRtDescriptorSet();
  helloVk.createRtPipeline();
  helloVk.createRtShaderBindingTable();

  bool useRayTracer = true;
  bool saveRender   = true;
  int  counter      = 60;
  // helloVk.m_pcRay.rho = 1.0f;
  // helloVk.m_pcRay.rho = 1.5f;
  // helloVk.m_pcRay.rho = 2.0f;
  // helloVk.m_pcRay.rho = 2.5f;
  // helloVk.m_pcRay.rho = 3.0f;
  // helloVk.m_pcRay.rho = 3.5f;
  helloVk.m_pcRay.rho = 4.0f;
  // helloVk.m_pcRay.rho = 4.5f;
  // helloVk.m_pcRay.rho = 5.0f;
  // helloVk.m_pcRay.rho = 5.5f;
  // helloVk.m_pcRay.rho = 6.0f;
  // helloVk.m_pcRay.rho = 6.5f;
  // helloVk.m_pcRay.rho = 7.0f;
  // helloVk.m_pcRay.rho = 7.5f;
  // helloVk.m_pcRay.rho = 8.0f;
  // helloVk.m_pcRay.rho = 8.5f;
  // helloVk.m_pcRay.rho = 9.0f;
  // helloVk.m_pcRay.rho = 9.5f;
  // helloVk.m_pcRay.rho = 10.0f;

  helloVk.createPostDescriptor();
  helloVk.createPostPipeline();
  helloVk.updatePostDescriptorSet();
  nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);


  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(helloVk.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show UI window.
    if(helloVk.showGui())
    {
      ImGuiH::Panel::Begin();
      ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
      ImGui::Checkbox("Ray tracer mode", &useRayTracer);    // we render a checkbox in the UI to switch between raytracing and resterization
      renderUI(helloVk);
      ImGui::SliderInt("Max Depth", &helloVk.m_pcRay.maxDepth, 1, 50);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
      ImGuiH::Panel::End();
    }

    // Writing avg framerate on file
    /*out << ImGui::GetIO().Framerate;
    out << "\n";*/

    CameraManip.updateAnim();

    // Change position of the subject if needed
    helloVk.updateSubjectPosition(); // ok

    // Start rendering the scene
    helloVk.prepareFrame();

    // Start command buffer of this frame
    auto                   curFrame = helloVk.getCurFrame();
    const VkCommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Updating camera buffer
    helloVk.updateUniformBuffer(cmdBuf);

    // Clearing screen
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen render pass
    {
      VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      offscreenRenderPassBeginInfo.clearValueCount = 2;
      offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
      offscreenRenderPassBeginInfo.renderPass      = helloVk.m_offscreenRenderPass;
      offscreenRenderPassBeginInfo.framebuffer     = helloVk.m_offscreenFramebuffer;
      offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Second scene light
      /* helloVk.m_pcRaster.lightPosition = vec3(-12.0f, 15.0f, 8.0f);
      helloVk.m_pcRaster.lightIntensity = 150.0f;*/

      // Third scene light
      /* helloVk.m_pcRaster.lightPosition = vec3(12.0f, 15.0f, -40.0f);
      helloVk.m_pcRaster.lightIntensity = 150.0f;*/

      if(counter == 60)
      {
        helloVk.m_pcRay.rho = helloVk.m_pcRay.rho + 0.5f;
        saveRender          = true;
      }

      // Rendering Scene
      // Note that raytracing behaves more like a compute task rather than graphics, so it doesn't need a render pass
      if(useRayTracer)
      {
        helloVk.raytrace(cmdBuf, clearColor);
      }
      else
      {
        vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        helloVk.rasterize(cmdBuf);
        vkCmdEndRenderPass(cmdBuf);
      }
    }


    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = helloVk.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = helloVk.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering tonemapper
      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      helloVk.drawPost(cmdBuf);
      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    if(saveRender)
    {
      helloVk.copyRenderedPosition(cmdBuf);
      helloVk.copyColorImage(cmdBuf);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    helloVk.submitFrame();
    // Presenting frame
    helloVk.presentFrame();

    if(saveRender)
    {
      helloVk.writeRenderedPosition(CURR_DIRECTORY);
      // helloVk.writeRenderedRays(CURR_DIRECTORY);
      helloVk.writeColorImage(CURR_DIRECTORY);
      saveRender = false;
      counter    = 0;
    }

    counter++;

    if(helloVk.m_pcRay.rho == 10.0f)
    {
      break;
    }
  }

  // out.close();

  // Cleanup
  vkDeviceWaitIdle(helloVk.getDevice());

  helloVk.destroyResources();
  helloVk.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
