# Removed Logic Ledger

This file records deliberately removed behavior that should not be restored as
generic backend code. If the product needs any of this again, describe it in a
domain skill, typed tool contract, or MCP/domain extension.

## 2026-06-29 Architecture Cleanup

### Legacy Streamlit UI

Removed files:

- `app.py`
- `components/`
- `utils/`
- `templates/`

Removed behavior:

- Streamlit chat/dashboard/sidebar entrypoint.
- Streamlit session parameter manager.
- Streamlit export helpers for old dashboard and chat reports.
- Streamlit data loader that expected SQLAlchemy.
- ydata-profiling Streamlit UI.
- References to the missing `agent.agent.Agent` package.
- References to the old `agent.artifact` model.

Replacement path:

- Supported UI is the React frontend.
- Supported runtime entrypoint is the FastAPI backend.
- Any future lightweight UI must call the backend API instead of importing agent
  internals.

### Investment SQL Table Routing

Removed behavior from `backend/data_access/sql_table_service.py`:

- Investment question detector.
- Primary-table overrides for `instrument_snapshot`, `price_history`, and
  `news_impact`.
- Extra-table selection that automatically attached investment tables.
- Finance-specific table scoring using ticker, sector, yield, dividend, risk,
  portfolio, issuer, price, news, sentiment, and related tokens.
- Special scoring for market movement and sector-comparison prompts.

Replacement path:

- Generic SQL table selection remains based on explicit table mentions, column
  mentions, schema/meta filtering, and join-related tables.
- Investment-specific routing must be expressed through `skills/`, typed tool
  contracts, MCP/domain extension manifests, or prompt instructions owned by the
  investment domain.
