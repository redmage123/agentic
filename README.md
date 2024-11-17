# Multi-Agent AI Architecture Demonstration
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

## Overview
This project demonstrates a multi-agent AI architecture through the implementation of a time series analysis system. It showcases different neural network paradigms, agent coordination patterns, and modern microservices architecture. The system integrates physics-inspired neural networks with generative AI to illustrate various approaches to AI system design.

### Educational Purpose
- Demonstrate different neural network architectures working together
- Showcase multi-agent system coordination
- Illustrate modern microservices patterns
- Present real-world AI model collaboration strategies

## Architecture

```mermaid
graph LR
    A[React Frontend] --> B[Flask Backend]
    B --> C[Traffic Control Agent]
    C --> D[Hamiltonian NN]
    C --> E[Fourier NN]
    C --> F[Perturbation NN]
    C --> G[Generative LLM]
```

### Components
- **Frontend**: React-based dashboard for visualization and interaction
- **Backend**: Flask service with REST API
- **Traffic Control Agent**: Orchestrates multiple AI agents via gRPC
- **AI Agents**:
  - Hamiltonian Neural Network: Conservation law modeling
  - Fourier Neural Network: Frequency domain analysis
  - Perturbation Theory Neural Network: Regime change detection
  - Generative LLM: Natural language integration and explanation

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 14+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Set up Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd services/client_service/frontend
npm install
```

4. Build and run with Docker:
```bash
./setup.sh
docker-compose up
```
## Environment Setup

This project requires certain environment variables to be set. To configure:

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

## Usage

### Running the System
1. Start the services:
```bash
./run_app.sh
```

2. Access the dashboard:
```
http://localhost:3000
```

### Running Tests
```bash
./run_tests.sh
```

## Project Structure
```
├── services/
│   ├── client_service/     # Frontend and Backend
│   ├── tca_service/        # Traffic Control Agent
│   ├── hamiltonian_agent/  # Hamiltonian NN Service
│   ├── fourier_agent/      # Fourier NN Service
│   ├── perturbation_agent/ # Perturbation Theory NN
│   └── generative_agent/   # LLM Service
├── models/                 # Neural Network Implementations
├── tests/                  # Test Suite
├── utils/                  # Utility Functions
└── protos/                # gRPC Protocol Definitions
```

## Educational Components

### Neural Network Approaches
- **Hamiltonian NN**: Models system dynamics using energy conservation principles
- **Fourier NN**: Analyzes frequency components and periodic patterns
- **Perturbation NN**: Detects and analyzes system regime changes
- **Generative LLM**: Provides natural language analysis and explanations

### Agent Coordination Patterns
- Round-robin distribution
- Capability-based routing
- Confidence-weighted aggregation
- Ensemble methods

## Generative AI Agent Prompts

The system uses a structured, modular prompt system for the generative AI agent located in `services/generative_agent/prompts/`. This modular approach allows for:
- Better maintainability
- Focused, specific prompts
- Flexible prompt chaining
- Easier testing and validation

### Prompt Structure

``` mermaid
graph TB
    subgraph "Prompt System Architecture"
        A[Financial Analysis Base Prompt] --> B{Prompt Router}

        subgraph "Core Analysis Frameworks"
            C[Tree of Thoughts<br>Framework]
            D[Logic of Thoughts<br>Framework]
        end

        subgraph "Specialized Analysis"
            E[Pattern Recognition]
            F[Risk Evaluation]
            G[Market Context]
        end

        subgraph "Output Processing"
            H[Synthesis]
            I[Validation]
        end

        B --> C
        B --> D
        B --> E
        B --> F
        B --> G

        C --> H
        D --> H
        E --> H
        F --> H
        G --> H

        H --> I

        I --> J[Final Response]

        %% Prompt Chain Configurations
        subgraph "Chain Configurations"
            K[Default Chain]
            L[Pattern Analysis]
            M[Full Analysis]
        end

        B -.-> K
        B -.-> L
        B -.-> M

        class A,B,C,D,E,F,G,H,I primary
        class K,L,M secondary
    end

classDef primary fill:#f9f,stroke:#333,stroke-width:2px
classDef secondary fill:#bbf,stroke:#333,stroke-width:2px
```

``` mermaid
sequenceDiagram
    participant U as User Query
    participant B as Base Prompt
    participant F as Framework Prompts
    participant S as Specialized Prompts
    participant V as Validation
    participant R as Response

    U->>B: Financial Query
    activate B
    B->>F: Apply Analysis Frameworks
    activate F

    par Tree of Thoughts
        F->>F: Apply ToT Framework
    and Logic of Thoughts
        F->>F: Apply LoT Framework
    end

    F->>S: Route to Specialized Analysis
    activate S

    par Pattern Analysis
        S->>S: Pattern Recognition
    and Risk Analysis
        S->>S: Risk Evaluation
    and Market Analysis
        S->>S: Market Context
    end

    S->>V: Synthesize & Validate
    activate V
    V->>R: Generate Response
    deactivate V
    deactivate S
    deactivate F
    deactivate B
    ```

``` mermaid
flowchart TD
    subgraph "Prompt Composition"
        A[Base Prompt] --> B{Analysis Type}

        B -->|Risk Analysis| C[Risk Chain]
        B -->|Pattern Analysis| D[Pattern Chain]
        B -->|Full Analysis| E[Full Chain]

        subgraph "Risk Chain"
            C --> F1[Base]
            F1 --> F2[Risk Evaluation]
            F2 --> F3[Validation]
            F3 --> F4[Synthesis]
        end

        subgraph "Pattern Chain"
            D --> G1[Base]
            G1 --> G2[Pattern Recognition]
            G2 --> G3[Validation]
            G3 --> G4[Synthesis]
        end

        subgraph "Full Chain"
            E --> H1[Base]
            H1 --> H2[All Specialized]
            H2 --> H3[All Frameworks]
            H3 --> H4[Validation]
            H4 --> H5[Synthesis]
        end
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C,D,E fill:#bfb,stroke:#333,stroke-width:2px

   ``` 
#### Financial Analysis Prompts
Located in `services/generative_agent/prompts/financial_analysis/`:
- `base.prompt`: Core financial analyst identity and approach
- `pattern_recognition.prompt`: Specialized pattern analysis techniques
- `market_context.prompt`: Market analysis framework
- `risk_evaluation.prompt`: Risk analysis methodologies
- `tot_framework.prompt`: Tree of Thoughts reasoning framework
- `lot_framework.prompt`: Logic of Thoughts analysis structure
- `synthesis.prompt`: Results synthesis and communication
- `validation.prompt`: Data and conclusion validation

Each prompt is designed to:
- Focus on a specific aspect of financial analysis
- Maintain consistency in analysis approach
- Provide structured, actionable outputs
- Include validation and confidence measures

### Using Prompts

The prompts can be used individually or chained together based on the analysis needs. Common combinations are configured in `conf/components/generative/default.yaml`:

- Default Chain: Base analysis with structured reasoning
- Pattern Analysis: Focused on pattern recognition and validation
- Full Analysis: Comprehensive analysis using all prompt components

### Extending Prompts

To add new prompts:
1. Create new .prompt file in appropriate directory
2. Add prompt configuration to yaml config
3. Add prompt tests in test suite
4. Document prompt purpose and usage

### Prompt Development Guidelines

When creating or modifying prompts:
- Keep each prompt focused and specific
- Include clear input/output expectations
- Consider chain compatibility
- Add appropriate validation steps
- Document prompt purpose
- Include example usage

### Testing Prompts

Prompts are tested through:
- Unit tests for loading and validation
- Integration tests for prompt chaining
- Response quality validation
- Performance benchmarking

See `tests/test_generative_agent/test_prompts/` for test implementations.
## Development

### Adding New Agents
1. Create new directory under `services/`
2. Define gRPC protocol in `protos/`
3. Implement agent interface
4. Register with Traffic Control Agent

### Modifying the Frontend
1. Navigate to `services/client_service/frontend`
2. Make changes to React components
3. Test using `npm test`
4. Build using `npm run build`

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Future Enhancements
- Additional neural network architectures
- Enhanced visualization components
- More agent coordination patterns
- Extended educational documentation

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Neural network implementations inspired by physics-based modeling
- Multi-agent architecture patterns from distributed systems design
- Modern microservices practices

## Contact
Braun Brelin
braun.brelin@ai-elevate.ai

## Disclaimer
This project is for educational and demonstration purposes only. It is not intended for production use or real-world financial applications.
foo bar baz
