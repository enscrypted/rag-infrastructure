# Contributing to RAG Infrastructure Stack

Thank you for your interest in contributing to the RAG Infrastructure Stack! This guide will help you get started.

## Ways to Contribute

- **Bug Reports**: Found an issue? Please let us know!
- **Feature Requests**: Have an idea for improvement? We'd love to hear it!
- **Documentation**: Help improve our docs, tutorials, and examples
- **Code**: Fix bugs, add features, improve performance
- **Examples**: Share your RAG applications and use cases

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/enscrypted/rag-infrastructure.git
   cd rag-infrastructure
   ```

2. **Test the Deployment**
   ```bash
   # Deploy on a test system
   chmod +x scripts/deploy.sh
   sudo scripts/deploy.sh
   ```

3. **Verify Everything Works**
   ```bash
   # Test connectivity
   chmod +x scripts/setup-dns-and-proxy.sh
   ./scripts/setup-dns-and-proxy.sh your-ip
   
   # Run example
   cd examples/basic-rag
   python simple_rag.py
   ```

### Development Guidelines

#### Code Style
- **Shell Scripts**: Use `set -e`, proper error handling, clear comments
- **Python**: Follow PEP 8, include type hints where helpful
- **Documentation**: Clear, concise, with practical examples

#### Testing Your Changes
- Test deployment on clean systems when possible
- Verify all services start correctly
- Test examples and tutorials
- Check documentation accuracy

#### Documentation Standards
- Include code examples for all features
- Add troubleshooting sections for common issues
- Keep README files up to date
- Use clear, beginner-friendly language

## Types of Contributions

### 🐛 Bug Reports

**Before submitting:**
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather system information and logs

**Include in your report:**
- Operating system and version
- Docker and Docker Compose versions
- Complete error messages and logs
- Steps to reproduce the issue
- Expected vs. actual behavior

### ✨ Feature Requests

**Good feature requests include:**
- Clear use case and motivation
- Detailed description of proposed functionality
- Consideration of implementation approach
- Discussion of alternatives considered

### 📚 Documentation Improvements

**Areas that need help:**
- Service-specific guides (Neo4j, Langfuse, etc.)
- Advanced RAG techniques and patterns
- Production deployment guides
- Troubleshooting and FAQ sections
- Video tutorials and walkthroughs

### 🔧 Code Contributions

**High-Priority Areas:**
- Additional vector databases (Pinecone, Weaviate integration)
- Enhanced deployment options (Kubernetes, cloud platforms)
- Performance optimizations
- Security improvements
- Monitoring and observability tools
- Additional UI improvements

### 📝 Example Applications

**We welcome examples for:**
- Industry-specific RAG applications
- Multi-modal RAG (text, images, audio)
- Graph-enhanced RAG with Neo4j
- Real-time RAG systems
- RAG evaluation and testing
- Integration with popular frameworks

## Submission Process

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clear, focused commits
   - Include tests where applicable
   - Update documentation as needed

3. **Test Thoroughly**
   - Test deployment process
   - Verify examples work
   - Check documentation accuracy
   - Run any existing tests

4. **Submit Pull Request**
   - Use clear, descriptive title
   - Explain what your changes do and why
   - Reference any related issues
   - Include testing notes

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Example/tutorial addition

## Testing
- [ ] Tested deployment process
- [ ] Tested affected services
- [ ] Updated documentation
- [ ] Tested examples/tutorials

## Screenshots/Logs
(If applicable, add screenshots or log outputs)

## Checklist
- [ ] Self-review of code
- [ ] Clear commit messages
- [ ] No hardcoded personal information
- [ ] Documentation updated
- [ ] Examples tested
```

### Review Process

1. **Automated Checks** (when CI/CD is set up)
   - Code formatting and linting
   - Documentation link checking
   - Basic deployment tests

2. **Manual Review**
   - Code quality and style
   - Documentation clarity
   - Security considerations
   - Performance implications

3. **Community Feedback**
   - Testing by other contributors
   - Discussion of approaches
   - Suggestions for improvement

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** in all interactions
- **Be constructive** when providing feedback
- **Be patient** with newcomers
- **Be open** to different perspectives and approaches

### Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be recognized in the following ways:

- **Contributors File**: All contributors listed in project documentation
- **Changelog**: Significant contributions noted in release notes
- **Community Highlights**: Outstanding contributions featured in project updates

## Development Resources

### Useful Tools

- **Docker**: For containerization testing
- **MongoDB Compass**: For database development
- **Postman**: For API testing
- **VS Code**: Recommended editor with Docker extension

### Learning Resources

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Docker Documentation](https://docs.docker.com/)
- [RAG Techniques Overview](https://docs.ragas.io/)
- [Ollama Model Documentation](https://ollama.ai/docs)

### Project Architecture

```
rag-infrastructure/
├── scripts/          # Deployment and setup scripts
├── services/         # Service configurations and custom UIs
├── docs/            # Documentation (services, concepts, tutorials)
├── examples/        # Example applications
└── README.md        # Main project documentation
```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes to deployment or APIs
- **MINOR**: New features, new services, significant improvements
- **PATCH**: Bug fixes, documentation updates, minor improvements

### Release Checklist

- [ ] Test deployment on multiple platforms
- [ ] Update documentation
- [ ] Test all examples
- [ ] Update CHANGELOG
- [ ] Tag release
- [ ] Update README badges

## Questions?

- **General Questions**: Open a discussion
- **Bug Reports**: Create an issue
- **Security Issues**: Email maintainers directly
- **Feature Ideas**: Start with a discussion or issue

Thank you for contributing to the RAG Infrastructure Stack! 🚀