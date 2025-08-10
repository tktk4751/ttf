# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
必ず日本語で回答してください｡
This is TTF (Trading Technical Framework) - a comprehensive Python-based trading backtesting and optimization system designed for financial markets. The system follows modular design principles and implements a wide range of technical indicators, signals, and trading strategies.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using Pipenv
pipenv install

# Activate virtual environment
pipenv shell

# Install in development mode
pip install -e .
```

### Running Tests
```bash
# Run all tests (uses pytest if available, otherwise Python unittest)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_indicator.py

# Run a single test
python tests/test_specific_indicator.py
```

### Data Management
```bash
# Load data from config
python main.py backtest

# Test indicators with real data
python examples/efficiency_ratio_v2_example.py

# Run indicator charts
python visualization/frama_chart.py --config config.yaml
```

### Running Indicators and Analysis
```bash
# Run specific indicator tests
python test_frama_pricesource.py
python test_ultimate_volatility_state.py

# Generate visualizations
python visualization/frama_chart_example.py
python examples/cosmic_adaptive_channel_example.py
```

## Code Architecture

### Core System Design

The system follows a layered architecture with clear separation of concerns:

**Base Layer (indicators/)**
- `Indicator` base class defines the contract for all technical indicators
- `PriceSource` utility provides standardized price data calculation (close, hl2, hlc3, ohlc4)
- All indicators inherit from base class and implement `calculate()` method
- Extensive use of Numba JIT compilation for performance-critical calculations

**Signal Layer (signals/)**
- Modular signal generation system with entry, exit, direction, and filter signals
- Each signal type has its own interface and implementations
- Signals combine indicators to generate trading decisions

**Strategy Layer (strategies/)**
- Strategy implementations combine multiple signals
- Base strategy class provides common functionality
- Strategies generate final trading decisions

**Data Layer (data/)**
- `DataLoader` handles CSV and Binance data sources
- `DataProcessor` normalizes and validates market data
- Support for multiple timeframes and symbols

### Key Design Patterns

**Indicator Architecture:**
- All indicators extend `indicators.indicator.Indicator`
- Consistent interface: `calculate(data) -> result`
- PriceSource integration for standardized price calculations
- Caching mechanisms for performance optimization
- Comprehensive error handling and validation

**Signal System:**
- Interface-based design in `signals/interfaces/`
- Implementation directory structure: `signals/implementations/{indicator_name}/`
- Each signal type (entry, exit, direction, filter) has dedicated modules

**Configuration-Driven:**
- Central `config.yaml` controls all system parameters
- Modular configuration sections for different components
- Support for multiple data sources and timeframes

### Important Modules

**indicators/price_source.py:**
Standardized price source calculations used throughout the system. Always use `PriceSource.calculate_source(data, src_type)` instead of manual price calculations.

**indicators/indicator.py:**
Base class for all indicators. Provides logging, validation, and common functionality. All custom indicators must inherit from this.

**data/binance_data_source.py:**
Primary data source for cryptocurrency market data. Handles data loading and validation.

**visualization/ directory:**
Comprehensive charting system using matplotlib and mplfinance. Each major indicator has its own chart implementation.

## Important Implementation Notes

### When Working with Indicators

1. **Always use PriceSource:** Instead of manual price calculations, use `PriceSource.calculate_source(data, src_type)` for consistency
2. **Inherit from Indicator base class:** All indicators must extend `indicators.indicator.Indicator`
3. **Implement caching:** Use caching patterns found in existing indicators for performance
4. **Use Numba for core calculations:** Performance-critical calculations should use `@njit` decorators
5. **Follow error handling patterns:** Implement try-catch blocks and return appropriate error states

### Data Handling Standards

- DataFrame structure: expects 'open', 'high', 'low', 'close', 'volume' columns
- NumPy array structure: OHLCV format with shape (n, 5) minimum
- Always validate data presence and format before processing
- Use float64 for numerical calculations to ensure precision

### Configuration System

The `config.yaml` file drives all system behavior:
- Data sources and symbols
- Indicator parameters  
- Backtest settings
- Optimization parameters
- Output configurations

When adding new functionality, extend the configuration schema appropriately.

### Testing Approach

- Each indicator should have a corresponding test file
- Tests should validate calculation accuracy and edge cases
- Use real market data for integration tests
- Performance tests for computational-heavy indicators

### Naming Conventions

- Indicators: PascalCase class names (e.g., `FRAMA`, `EfficiencyRatio`)
- Files: snake_case (e.g., `ultimate_volatility_state.py`)
- Methods: snake_case (e.g., `calculate_frama_core`)
- Constants: UPPER_SNAKE_CASE

### Performance Considerations

- Use NumPy arrays for all numerical computations
- Implement Numba JIT compilation for core calculation functions
- Cache results when appropriate to avoid recalculation
- Profile performance-critical sections and optimize bottlenecks

## Key Dependencies

- **NumPy/Pandas:** Core data manipulation
- **Numba:** JIT compilation for performance
- **PyYAML:** Configuration file parsing
- **Matplotlib/mplfinance:** Visualization
- **Optuna:** Parameter optimization (in optimization modules)

## File Structure Patterns

```
indicators/
├── indicator.py              # Base indicator class
├── price_source.py          # Standardized price calculations  
├── {indicator_name}.py      # Individual indicator implementations
├── smoother/               # Smoothing indicators (FRAMA, etc.)
└── examples/               # Usage examples

signals/implementations/{indicator}/
├── entry.py               # Entry signal implementation
├── exit.py                # Exit signal implementation  
├── direction.py           # Direction signal implementation
└── filter.py              # Filter signal implementation

visualization/
├── {indicator}_chart.py   # Chart implementation for indicator
└── charts/                # Reusable chart components
```

This modular structure enables easy extension and maintenance of the trading system components.