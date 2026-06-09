"""Alembic migration environment for rfsn_v11.

Reads the database URL from the ``RFSN_DB_URL`` environment variable,
falling back to ``sqlalchemy.url`` in ``alembic.ini``.

Supports both *offline* (SQL-script generation) and *online*
(live-database) migration modes.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Alembic Config object — gives access to alembic.ini values.
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import the package metadata so Alembic can autogenerate migrations.
# ---------------------------------------------------------------------------
from rfsn_v11.db import metadata as target_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# Resolve the database URL.
# Environment variable RFSN_DB_URL takes priority over alembic.ini.
# ---------------------------------------------------------------------------
_db_url = os.environ.get("RFSN_DB_URL") or config.get_main_option("sqlalchemy.url")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)


# ===========================================================================
# Offline migration — generate SQL script without a live database connection.
# ===========================================================================

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine; calls to
    context.execute() emit SQL to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ===========================================================================
# Online migration — connect to the database and apply migrations directly.
# ===========================================================================

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine, associates a connection with the context, and then
    runs the migrations inside a transaction.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry-point: choose online or offline based on Alembic context state.
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
