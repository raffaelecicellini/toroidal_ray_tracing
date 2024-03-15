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


#include <sstream>
#include <limits>


#define STB_IMAGE_IMPLEMENTATION
#include "obj_loader.h"
#include "stb_image.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  const auto&    proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = nvmath::invert(view);
  hostUBO.projInverse = nvmath::invert(proj);

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  // auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT);
  // Points
  m_descSetLayoutBind.addBinding(SceneBindings::ePoint, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiCloudData{m_cloudDataBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::ePoint, &dbiCloudData));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};  // not needed?

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);

  // Create vertex and fragment shader stages
  VkShaderModule vertShaderModule =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/vert_shader.vert.spv", true, defaultSearchPaths, true));
  VkShaderModule fragShaderModule =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/frag_shader.frag.spv", true, defaultSearchPaths, true));

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName  = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName  = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};


  // Input assembly state
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  // Create the vertex input state info structure
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount   = 0;  // No vertex binding descriptions
  vertexInputInfo.vertexAttributeDescriptionCount = 0; 

  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

  // Dynamic viewport and scissor
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates    = dynamicStates.data();
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount  = 1;

  // Rasterization state
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_POINT;
  rasterizer.lineWidth               = 2.5f;
  rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;

  // Multisample state
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable  = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Color blend state
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable     = VK_FALSE;
  colorBlending.logicOp           = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount   = 1;
  colorBlending.pAttachments      = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  // Depth test
  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable  = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable     = VK_FALSE;

  // Create graphics pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount          = 2;
  pipelineInfo.pStages             = shaderStages;
  pipelineInfo.pVertexInputState   = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState      = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState   = &multisampling;
  pipelineInfo.pColorBlendState    = &colorBlending;
  pipelineInfo.pDynamicState       = &dynamicState;
  pipelineInfo.pDepthStencilState  = &depthStencil;
  pipelineInfo.layout              = m_pipelineLayout;
  pipelineInfo.renderPass          = m_offscreenRenderPass;
  pipelineInfo.subpass             = 0;
  pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

  if(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
  vkDestroyShaderModule(m_device, vertShaderModule, nullptr);

  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_cloudDataBuffer);
  m_alloc.destroy(m_cloudColorImage);

  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all points with a single call
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

  uint32_t numPointsToDraw = m_cloudData.size();
  vkCmdDraw(cmdBuf, numPointsToDraw, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }


  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writeDescriptorSets = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);


  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Instead of having at the end of submitFrame() we have it here so that we can copy the image to txt file
//
void HelloVulkan::presentFrame()
{
  // Presenting frame
  m_swapChain.present(m_queue);
}

//--------------------------------------------------------------------------------------------------
// Load data points from txt files
//
void HelloVulkan::loadPoints(const char* dir) {
  // Load points position
  std::string filename;
  // filename = "data/input/renderedPosition1.000000.txt";
  // filename = "data/input/renderedPosition1.500000.txt";
  // filename = "data/input/renderedPosition2.000000.txt";
  // filename = "data/input/renderedPosition2.500000.txt";
  // filename = "data/input/renderedPosition3.000000.txt";
  // filename = "data/input/renderedPosition3.500000.txt";
  // filename = "data/input/renderedPosition4.000000.txt";
  // filename = "data/input/renderedPosition4.500000.txt";
  // filename = "data/input/renderedPosition5.000000.txt";
  // filename = "data/input/renderedPosition5.500000.txt";
  // filename = "data/input/renderedPosition6.000000.txt";
  // filename = "data/input/renderedPosition6.500000.txt";
  // filename = "data/input/renderedPosition7.000000.txt";
  // filename = "data/input/renderedPosition7.500000.txt";
  // filename = "data/input/renderedPosition8.000000.txt";
  // filename = "data/input/renderedPosition8.500000.txt";
  // filename = "data/input/renderedPosition9.000000.txt";
  // filename = "data/input/renderedPosition9.500000.txt";
  // filename = "data/input/renderedPosition10.000000.txt";
  
  filename = "data/first_version/simple0Position.txt";
  // filename = "data/first_version/simple1Position.txt";
  // filename = "data/first_version/simple2Position.txt";
  // filename = "data/first_version/city0Position.txt";
  // filename = "data/first_version/city1Position.txt";
  // filename = "data/first_version/city2Position.txt";
  // filename = "data/first_version/house0Position.txt";
  // filename = "data/first_version/house1Position.txt";
  // filename = "data/first_version/house2Position.txt";

  filename = dir + filename;
  std::ifstream inputPos(filename);

  std::string line;
  while(std::getline(inputPos, line))
  {
    nvmath::vec3f point;
    std::istringstream iss(line);
    std::string        xStr, yStr, zStr;
    bool               xNaN, yNaN, zNaN;

    if(iss >> xStr >> yStr >> zStr)
    {
      xNaN = xStr.find("-nan") != std::string::npos;
      yNaN = yStr.find("-nan") != std::string::npos;
      zNaN = zStr.find("-nan") != std::string::npos;

      // Convert strings to floats, handling NaN if necessary
      point.x = xNaN ? std::numeric_limits<float>::lowest() : std::stof(xStr);
      point.y = yNaN ? std::numeric_limits<float>::lowest() : std::stof(yStr);
      point.z = zNaN ? std::numeric_limits<float>::lowest() : std::stof(zStr);
    }
    else
    {
      // Error reading values, assign lowest value to all components
      point.x = std::numeric_limits<float>::lowest();
      point.y = std::numeric_limits<float>::lowest();
      point.z = std::numeric_limits<float>::lowest();
    }

    m_positions.push_back(point);
  }

  inputPos.close();

  // Load points color
  // filename = "data/output/simple0_renderedColor1.000000.txt";
  // filename = "data/output/simple0_renderedColor1.500000.txt";
  // filename = "data/output/simple0_renderedColor2.000000.txt";
  // filename = "data/output/simple0_renderedColor2.500000.txt";
  // filename = "data/output/simple0_renderedColor3.000000.txt";
  // filename = "data/output/simple0_renderedColor3.500000.txt";
  // filename = "data/output/simple0_renderedColor4.000000.txt";
  // filename = "data/output/simple0_renderedColor4.500000.txt";
  // filename = "data/output/simple0_renderedColor5.000000.txt";
  // filename = "data/output/simple0_renderedColor5.500000.txt";
  // filename = "data/output/simple0_renderedColor6.000000.txt";
  // filename = "data/output/simple0_renderedColor6.500000.txt";
  // filename = "data/output/simple0_renderedColor7.000000.txt";
  // filename = "data/output/simple0_renderedColor7.500000.txt";
  // filename = "data/output/simple0_renderedColor8.000000.txt";
  // filename = "data/output/simple0_renderedColor8.500000.txt";
  // filename = "data/output/simple0_renderedColor9.000000.txt";
  // filename = "data/output/simple0_renderedColor9.500000.txt";
  // filename = "data/output/simple0_renderedColor10.000000.txt";

  filename = "data/first_version/simple0Color.txt";
  // filename = "data/first_version/simple1Color.txt";
  // filename = "data/first_version/simple2Color.txt";
  // filename = "data/first_version/city0Color.txt";
  // filename = "data/first_version/city1Color.txt";
  // filename = "data/first_version/city2Color.txt";
  // filename = "data/first_version/house0Color.txt";
  // filename = "data/first_version/house1Color.txt";
  // filename = "data/first_version/house2Color.txt";

  filename             = dir + filename;
  std::ifstream inputCol(filename);

  while(std::getline(inputCol, line))
  {
    nvmath::vec3f      color;
    std::istringstream iss(line);
    std::string xStr, yStr, zStr;
    bool        xNaN, yNaN, zNaN;

    if(iss >> xStr >> yStr >> zStr)
    {
      xNaN = xStr.find("-nan") != std::string::npos;
      yNaN = yStr.find("-nan") != std::string::npos;
      zNaN = zStr.find("-nan") != std::string::npos;

      // Convert strings to floats, handling NaN if necessary
      color.x = xNaN ? std::numeric_limits<float>::lowest() : std::stof(xStr);
      color.y = yNaN ? std::numeric_limits<float>::lowest() : std::stof(yStr);
      color.z = zNaN ? std::numeric_limits<float>::lowest() : std::stof(zStr);
    }
    else
    {
      // Error reading values, assign lowest value to all components
      color.x = std::numeric_limits<float>::lowest();
      color.y = std::numeric_limits<float>::lowest();
      color.z = std::numeric_limits<float>::lowest();
    }

    m_colors.push_back(color);
  }

  inputCol.close();
}

//--------------------------------------------------------------------------------------------------
// Create a buffer as big as a vector of Points
//
void HelloVulkan::createCloudDataBuffer() 
{
  // Check if the 2 vectors are equal in length
  if(m_positions.size() != m_colors.size())
  {
    throw std::runtime_error("Number of positions and colors don't match!");
  }

  // Create a vector of Points
  for(int i = 0; i < m_positions.size(); i++)
  {
    Point point;

    point.pos = nvmath::vec4f(m_positions[i], 0);
    point.color = nvmath::vec4f(m_colors[i], 0);

    m_cloudData.push_back(point);
  }

  // Create buffer to hold the vector of Points
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_cloudDataBuffer  = m_alloc.createBuffer(cmdBuf, m_cloudData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_cloudDataBuffer.buffer, "CloudDataBuffer");
}

//--------------------------------------------------------------------------------------------------
// Writing color of each pixel of rendered image
//
void HelloVulkan::copyColorImage(const VkCommandBuffer& cmdBuf)
{
  // Create the linear tiled destination image to copy to and to read the memory from
  VkImageCreateInfo imgCreateInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imgCreateInfo.imageType         = VK_IMAGE_TYPE_2D;
  imgCreateInfo.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
  imgCreateInfo.extent.width      = m_size.width;
  imgCreateInfo.extent.height     = m_size.height;
  imgCreateInfo.extent.depth      = 1;
  imgCreateInfo.arrayLayers       = 1;
  imgCreateInfo.mipLevels         = 1;
  imgCreateInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
  imgCreateInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  imgCreateInfo.tiling            = VK_IMAGE_TILING_LINEAR;
  imgCreateInfo.usage             = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  // Create the image
  m_cloudColorImage = m_alloc.createImage(imgCreateInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Pipeline barrier for both images
  VkImageMemoryBarrier barrier{};
  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.image                           = m_offscreenColor.image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;

  VkPipelineStageFlags sourceStage      = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  VkImageMemoryBarrier barrierDst{};
  barrierDst.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrierDst.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
  barrierDst.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrierDst.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrierDst.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrierDst.image                           = m_cloudColorImage.image;
  barrierDst.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrierDst.subresourceRange.baseMipLevel   = 0;
  barrierDst.subresourceRange.levelCount     = 1;
  barrierDst.subresourceRange.baseArrayLayer = 0;
  barrierDst.subresourceRange.layerCount     = 1;
  barrierDst.srcAccessMask                   = 0;
  barrierDst.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;

  sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
  destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrierDst);

  // Copying the image
  VkImageCopy imageCopyRegion{};
  imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopyRegion.srcSubresource.layerCount = 1;
  imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopyRegion.dstSubresource.layerCount = 1;
  imageCopyRegion.extent.width              = m_size.width;
  imageCopyRegion.extent.height             = m_size.height;
  imageCopyRegion.extent.depth              = 1;

  vkCmdCopyImage(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_cloudColorImage.image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);

  // Pipeline barrier to transition back the layouts
  barrierDst.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrierDst.oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrierDst.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
  barrierDst.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrierDst.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrierDst.image                           = m_cloudColorImage.image;
  barrierDst.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrierDst.subresourceRange.baseMipLevel   = 0;
  barrierDst.subresourceRange.levelCount     = 1;
  barrierDst.subresourceRange.baseArrayLayer = 0;
  barrierDst.subresourceRange.layerCount     = 1;
  barrierDst.srcAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrierDst.dstAccessMask                   = VK_ACCESS_MEMORY_READ_BIT;

  sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
  destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrierDst);

  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.image                           = m_offscreenColor.image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;

  sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
  destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

//--------------------------------------------------------------------------------------------------
// Copy the color data to txt file
//
void HelloVulkan::writeColorImage(const char* dir)
{
  const char* imagedata;
  // Get layout of the image (including row pitch)
  VkImageSubresource subResource{};
  subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  VkSubresourceLayout subResourceLayout;
  VkDeviceMemory      imageMemory = m_alloc.getDMA()->getMemoryInfo(m_cloudColorImage.memHandle).memory;

  vkGetImageSubresourceLayout(m_device, m_cloudColorImage.image, &subResource, &subResourceLayout);

  // Map image memory so we can start copying from it
  vkMapMemory(m_device, imageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&imagedata);
  imagedata += subResourceLayout.offset;

  std::string filename;
  // filename = "data/simple0ptCloudImage_10.txt";
  // filename = "data/simple1ptCloudImage_10.txt";
  // filename = "data/simple2ptCloudImage_10.txt";
  // filename = "data/city0ptCloudImage_10.txt";
  // filename = "data/city1ptCloudImage_10.txt";
  // filename = "data/city2ptCloudImage_10.txt";
  // filename = "data/house0ptCloudImage_10.txt";
  // filename = "data/house1ptCloudImage_10.txt";
  filename = "data/house2ptCloudImage_10.txt";
  filename = dir + filename;
  std::ofstream outfile(filename);
  // Iterate over each row and column of pixels
  for(uint32_t y = 0; y < m_size.height; ++y)
  {
    for(uint32_t x = 0; x < m_size.width; ++x)
    {
      // Calculate the memory offset for the current pixel
      size_t pixelOffset = y * subResourceLayout.rowPitch + x * sizeof(float) * 4;

      // Read the color values at the offset
      float* pixelColor = (float*)(imagedata + pixelOffset);

      // Write the color values to the file
      outfile << pixelColor[0] << " " << pixelColor[1] << " " << pixelColor[2] << std::endl;
    }
  }
  outfile.close();

  vkUnmapMemory(m_device, imageMemory);
}