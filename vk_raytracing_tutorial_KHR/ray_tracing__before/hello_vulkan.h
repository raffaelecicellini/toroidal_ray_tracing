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

#pragma once

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"
// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvkhl::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  struct ObjInstance
  {
    nvmath::mat4f transform;    // Matrix of the instance
    uint32_t      objIndex{0};  // Model index reference
  };


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1},                // Identity matrix
      {10.f, 15.f, 8.f},  // light position
      0,                  // instance Id
      100.f,              // light intensity
      0                   // light type
  };

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene


  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

  // #VKRay
  void initRayTracing();                              // function to start the raytracing process
  auto objectToVkGeometryKHR(const ObjModel& model);  // method to convert geometry data of an ObjModel into a set of structures (held into nvvk::RayTracingBuilderKHR::BlasInput) consumed by AS builder
  void createBottomLevelAS();  // function to generate a BlasInput for each object and trigger a BLAS build
  void createTopLevelAS();  // function to create (triggered after creation of all object instances) a TLAS that contains all the instances: in an instance we have transform matrix, id of the corresponding BLAS, instance id, index of hit group containing the shaders to call upon hitting
  void createRtDescriptorSet();  // we reuse the descriptor set of rasterization + we use a new one to reference TLAS and buffer to hold output image
  void updateRtDescriptorSet();  // as for rasterization, also rt desc set needs to be updated if its contents change (i.e. when resizing the windows since this triggers recreation of uotput image)
  void createRtPipeline();       // function to create the ray tracing pipeline
  void createRtShaderBindingTable();                                              // function to create the SBT
  void raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor /*, const float y*/);  // function that will record commands to call raytracing shaders
  void updateSubjectPosition();
  void copyRenderedPosition(const VkCommandBuffer& cmdBuf);
  void copyColorImage(const VkCommandBuffer& cmdBuf);
  void writeRenderedPosition(const char* dir);
  void writeRenderedRays(const char* dir);
  void writeColorImage(const char* dir);
  void presentFrame();

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};  // this member stores the GPU capabilities for rt
  nvvk::RaytracingBuilderKHR m_rtBuilder;  // helper class, acts as a container of a TLAS with an array of BLASes with utility functions to build acc structs
  nvvk::DescriptorSetBindings m_rtDescSetLayoutBind;  // we are defining all objects needed for the additional descriptor set
  VkDescriptorPool                                  m_rtDescPool;
  VkDescriptorSetLayout                             m_rtDescSetLayout;
  VkDescriptorSet                                   m_rtDescSet;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;
  PushConstantRay                                   m_pcRay{{}, {}, 0, 0, 10 /*, 0 */ };  // push constants for ray tracer
  nvvk::Buffer                                      m_rtSBTBuffer;
  VkStridedDeviceAddressRegionKHR                   m_rgenRegion{};
  VkStridedDeviceAddressRegionKHR                   m_missRegion{};
  VkStridedDeviceAddressRegionKHR                   m_hitRegion{};
  VkStridedDeviceAddressRegionKHR                   m_callRegion{};
  std::vector<VkAccelerationStructureInstanceKHR>    m_tlas;
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> m_blas;
  nvvk::Buffer                                       m_rtRenderedBuffer;
  nvvk::Buffer                                       m_rtRDataBuffer;
  nvvk::Image                                        m_rtColorImage;
};
