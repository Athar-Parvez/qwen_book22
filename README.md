# Physical AI & Humanoid Robotics Textbook

A comprehensive educational platform for teaching Physical AI and Humanoid Robotics, combining theoretical knowledge with practical applications.

## 📚 Overview

This project creates a complete textbook for teaching Physical AI & Humanoid Robotics using modern web technologies. The platform bridges the digital brain (AI) with the physical body (robots) by providing:

- Interactive textbook content covering Physical AI and Embodied Intelligence
- Integration with robotics simulation frameworks like ROS 2, Gazebo, and NVIDIA Isaac
- Vision-Language-Action (VLA) capabilities for real-world interaction
- RAG-based chatbot for interactive learning support

## 🏗️ Architecture

### Frontend
- **Framework**: Docusaurus
- **Features**: Modern, responsive UI with dark/light mode, textbook content, embedded chatbot
- **Location**: `frontend/docusaurus/`
- **Documentation**: See `frontend/docusaurus/README.md`

### Backend
- **Framework**: FastAPI 
- **Features**: RAG functionality, chatbot API, content management, vector search
- **Location**: `backend/fastapi/`
- **Documentation**: See `backend/fastapi/README.md`

## 🚀 Features

- **Modular Content Architecture**: Each chapter stands alone yet connects to broader concepts
- **Technology Integration First**: Real implementations alongside theoretical understanding
- **Simulation-First Approach**: Concepts validated through simulation before real hardware
- **Embodied Intelligence Focus**: Bridge between AI minds and robotic bodies
- **Production-Ready Education**: Industry-standard code examples and practices

## 🛠️ Chapters

The textbook covers:

1. **Introduction to Physical AI & Embodied Intelligence** - Core concepts and foundations
2. **The Robotic Nervous System (ROS 2)** - Communication and control frameworks
3. **The Digital Twin** - Simulation and modeling environments  
4. **The AI-Robot Brain (NVIDIA Isaac™)** - Advanced AI integration for robotics
5. **Vision-Language-Action (VLA)** - Perception and action in robotics
6. **Capstone Project – Autonomous Humanoid** - Complete implementation project

## 🤖 Technologies Used

- **Docusaurus**: For textbook content delivery
- **FastAPI**: For backend services
- **Python**: Primary implementation language
- **OpenAI Agents SDK**: For AI integration
- **Qdrant Cloud**: Vector storage for RAG functionality
- **Neon Serverless Postgres**: Metadata storage
- **ROS 2**: Robotics middleware
- **NVIDIA Isaac**: Photorealistic simulation
- **GitHub Pages**: Deployment platform

## 📖 Getting Started

1. **Frontend Setup** (See `frontend/docusaurus/README.md` for detailed instructions):
   ```bash
   cd frontend/docusaurus
   npm install
   npm start
   ```

2. **Backend Setup** (See `backend/fastapi/README.md` for detailed instructions):
   ```bash
   cd backend/fastapi
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Content Development**:
   - Add textbook content in `frontend/docusaurus/docs/`
   - Update sidebar navigation in `frontend/docusaurus/sidebars.js`
   - Run ingestion script to update RAG system

## 🤝 Contributing

We welcome contributions to enhance the textbook content and platform features. Please see our contributing guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) (to be created).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions about this project, please open an issue in the repository.