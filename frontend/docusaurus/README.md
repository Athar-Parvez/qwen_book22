# Frontend – Docusaurus

This folder contains the Docusaurus-based textbook UI.

## Features:
- Markdown-based chapters
- Dark/Light mode
- Sidebar navigation
- Embedded RAG chatbot (planned)

## Status: In progress

## Getting Started

To run the frontend locally:

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open your browser to [http://localhost:3000](http://localhost:3000) to view the textbook.

## Project Structure

```
frontend/docusaurus/
├── docs/                 # Textbook content in Markdown format
├── src/
│   ├── components/       # Custom React components
│   ├── css/              # Custom styles
│   └── pages/            # Additional pages
├── static/               # Static assets (images, etc.)
├── docusaurus.config.js  # Docusaurus configuration
├── sidebars.js          # Navigation sidebar configuration
└── package.json         # Dependencies and scripts
```

## Documentation Structure

- Textbook chapters are stored as Markdown files in the `docs/` directory
- Navigation is configured in `sidebars.js`
- Custom components can be added to extend functionality
- Static assets (images, diagrams) go in the `static/` directory

## Theming

- Dark/light mode is implemented using Docusaurus' built-in theme system
- Custom CSS can be added in `src/css/custom.css`
- Theme configurations are in `docusaurus.config.js`

## Planned Features

- [ ] Integration with RAG chatbot API
- [ ] Interactive code examples
- [ ] Search functionality
- [ ] User progress tracking

## Contributing

To add new textbook content:
1. Create a new Markdown file in the `docs/` directory
2. Add an entry to `sidebars.js` to include it in the navigation
3. Follow the existing content structure and formatting

For custom components or styling:
1. Create React components in `src/components/`
2. Add custom CSS in `src/css/custom.css`