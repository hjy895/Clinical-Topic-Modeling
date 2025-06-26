# Contributing to Clinical Topic Modeling Framework

We welcome contributions to the Clinical Topic Modeling Framework! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports**: Report issues you encounter
- **✨ Feature Requests**: Suggest new functionality
- **📝 Documentation**: Improve or add documentation
- **🔧 Code Contributions**: Fix bugs or add features
- **📊 Examples**: Add new examples or use cases
- **🧪 Tests**: Improve test coverage

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/clinical-topic-modeling.git
   cd clinical-topic-modeling
   ```

2. **Create a development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🐛 Reporting Bugs

Before submitting a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Use the latest version** of the framework
3. **Provide a minimal reproducible example**

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data with '...'
2. Configure framework with '...'
3. Run evaluation with '...'
4. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.0]
- Framework version: [e.g. 1.0.0]
- Dependencies: [paste pip freeze output]

**Additional context**
Any other context about the problem.
```

## ✨ Feature Requests

We love feature requests! Please:

1. **Check existing issues** for similar requests
2. **Describe the problem** you're trying to solve
3. **Explain your proposed solution**
4. **Consider the scope** and impact

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Use case**
How would this feature be used? Who would benefit?

**Implementation ideas**
If you have ideas about implementation, share them here.
```

## 🔧 Code Contributions

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests**
   ```bash
   pytest tests/
   python -m pytest tests/ --cov=src/
   ```

4. **Run code quality checks**
   ```bash
   black src/ tests/ examples/
   flake8 src/ tests/ examples/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/amazing-feature
   ```

### Coding Standards

#### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting
- Use **type hints** for function parameters and return values
- Maximum line length: **88 characters** (Black default)

#### Code Structure

```python
"""
Module docstring describing the purpose and functionality.
"""

import standard_library_modules
import third_party_modules
from local_modules import local_classes

from typing import Dict, List, Optional, Any


class ExampleClass:
    """
    Class docstring with description.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self, data: List[str]) -> Dict[str, Any]:
        """
        Method docstring with description.
        
        Args:
            data: Input data description
            
        Returns:
            Dictionary with results
            
        Raises:
            ValueError: When data is invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        return {"result": "processed"}
```

#### Documentation Standards

- Use **Google-style docstrings**
- Include type hints in function signatures
- Document all parameters, return values, and exceptions
- Add examples for complex functions

#### Testing Standards

- Write tests for all new functionality
- Use **pytest** framework
- Aim for **>80% code coverage**
- Include both unit tests and integration tests

```python
import pytest
from src.framework import ClinicalTopicModelingFramework


def test_framework_initialization():
    """Test framework initializes correctly."""
    framework = ClinicalTopicModelingFramework()
    assert framework is not None
    assert framework.config is not None


def test_data_loading_with_sample_data():
    """Test framework can load sample data."""
    framework = ClinicalTopicModelingFramework()
    # Test implementation here
    pass


@pytest.mark.parametrize("input_data,expected", [
    ({"test": "data"}, True),
    ({}, False),
])
def test_parametrized_function(input_data, expected):
    """Test function with multiple parameter sets."""
    result = some_function(input_data)
    assert bool(result) == expected
```

### Commit Message Guidelines

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add XGBoost classifier support

Add XGBoost classifier to ensemble models with optimized hyperparameters
for clinical data characteristics.

Closes #123
```

```
fix(data): handle missing values in clinical topic extraction

- Add validation for NaN values in input data
- Implement fallback strategies for missing clinical measurements
- Add comprehensive tests for edge cases

Fixes #456
```

## 📝 Documentation Contributions

We appreciate documentation improvements! This includes:

- **API documentation**: Improving docstrings and type hints
- **User guides**: Adding tutorials and examples
- **Developer docs**: Contributing to development documentation
- **README improvements**: Making the project more accessible

### Documentation Build

To build documentation locally:

```bash
cd docs/
make html
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_framework"
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test speed and memory usage

### Test Data

- Use **synthetic data** for tests when possible
- Include **edge cases** and **error conditions**
- Ensure tests are **deterministic** (use fixed random seeds)

## 📊 Adding Examples

When adding examples:

1. **Create clear, focused examples** that demonstrate specific features
2. **Include comprehensive comments** explaining each step
3. **Use realistic but synthetic data** to protect privacy
4. **Test examples** to ensure they work correctly
5. **Update documentation** to reference new examples

### Example Structure

```python
"""
Example: [Brief Description]

This example demonstrates how to [specific functionality].
Use case: [when this would be useful]
"""

# Step 1: Setup
# Explanation of what we're setting up

# Step 2: Data preparation
# Explanation of data requirements

# Step 3: Model configuration
# Explanation of configuration choices

# Step 4: Execution
# Explanation of the main workflow

# Step 5: Results analysis
# Explanation of how to interpret results
```

## 🔍 Code Review Process

### For Contributors

1. **Self-review** your code before submitting
2. **Write descriptive PR descriptions** explaining changes
3. **Respond to feedback** constructively
4. **Update your PR** based on review comments

### Review Criteria

We evaluate contributions based on:

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it well-written and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Compatibility**: Does it break existing functionality?
- **Performance**: Does it introduce performance issues?

## 🚀 Release Process

### Versioning

We use **Semantic Versioning** (SemVer):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backward-compatible functionality additions
- **PATCH** version: Backward-compatible bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Release notes prepared

## 💬 Getting Help

If you need help with contributing:

1. **Check existing documentation** and examples
2. **Search existing issues** for similar questions
3. **Join our discussions** on GitHub Discussions
4. **Contact maintainers** via email or issue comments

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of:

- Age, body size, disability, ethnicity, gender identity
- Level of experience, nationality, personal appearance
- Race, religion, sexual identity and orientation

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Project maintainers have the right and responsibility to remove, edit, or reject:
- Comments, commits, code, wiki edits, issues
- Other contributions that are not aligned with this Code of Conduct

## 🙏 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** for major features
- **Project README** for ongoing contributors

## 📞 Contact

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [maintainer-email] for private matters

---

Thank you for contributing to the Clinical Topic Modeling Framework! 🎉

Your contributions help advance clinical AI research and improve patient care.