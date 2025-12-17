import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Creating a custom sidebar for the textbook
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Introduction to Physical AI & Embodied Intelligence',
      items: ['intro-physical-ai/overview', 'intro-physical-ai/concepts', 'intro-physical-ai/applications'],
    },
    {
      type: 'category',
      label: 'The Robotic Nervous System (ROS 2)',
      items: ['ros2-nervous-system/overview', 'ros2-nervous-system/nodes', 'ros2-nervous-system/topics', 'ros2-nervous-system/services'],
    },
    {
      type: 'category',
      label: 'The Digital Twin (Simulation)',
      items: ['digital-twin/overview', 'digital-twin/gazebo', 'digital-twin/physics-modeling', 'digital-twin/simulation-workflows'],
    },
    {
      type: 'category',
      label: 'The AI-Robot Brain (NVIDIA Isaac)',
      items: ['ai-brain/overview', 'ai-brain/isaac-ros', 'ai-brain/perception', 'ai-brain/planning', 'ai-brain/control'],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action (VLA)',
      items: ['vla/overview', 'vla/models', 'vla/integration', 'vla/applications'],
    },
    {
      type: 'category',
      label: 'Capstone: Autonomous Humanoid Project',
      items: ['capstone/overview', 'capstone/project-requirements', 'capstone/implementation', 'capstone/evaluation'],
    },
    'conclusion'
  ],
};

export default sidebars;
