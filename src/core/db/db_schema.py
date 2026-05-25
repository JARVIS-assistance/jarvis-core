from __future__ import annotations

from .db_connection import DBClient

POSTGRES_STATEMENTS = [
    # Enums
    """
    DO $$
    BEGIN
        CREATE TYPE user_status AS ENUM ('ACTIVE', 'INACTIVE', 'BANNED');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END$$;
    """,
    """
    DO $$
    BEGIN
        CREATE TYPE chat_status AS ENUM ('ACTIVE', 'ARCHIVED');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END$$;
    """,
    """
    DO $$
    BEGIN
        CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system', 'tool');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END$$;
    """,
    """
    DO $$
    BEGIN
        CREATE TYPE memory_type AS ENUM ('preference', 'fact', 'task');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END$$;
    """,
    """
    DO $$
    BEGIN
        CREATE TYPE ai_provider_mode AS ENUM ('token', 'local');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END$$;
    """,
    # Tables
    """
    CREATE TABLE IF NOT EXISTS users (
      id uuid PRIMARY KEY,
      email varchar(320) NOT NULL UNIQUE,
      name varchar(100),
      status user_status NOT NULL DEFAULT 'ACTIVE',
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS user_settings (
      user_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      locale varchar(20) NOT NULL DEFAULT 'ko-KR',
      timezone varchar(64) NOT NULL DEFAULT 'Asia/Seoul',
      response_style varchar(40) NOT NULL DEFAULT 'balanced',
      metadata jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS personas (
      id uuid PRIMARY KEY,
      owner_user_id uuid REFERENCES users(id) ON DELETE CASCADE,
      name varchar(80) NOT NULL,
      description text,
      prompt_template text NOT NULL,
      tone varchar(40),
      is_active boolean NOT NULL DEFAULT true
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS user_personas (
      id uuid PRIMARY KEY,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      persona_id uuid NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
      alias varchar(80)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chats (
      id uuid PRIMARY KEY,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      status chat_status NOT NULL DEFAULT 'ACTIVE',
      last_selected_user_persona_id uuid REFERENCES user_personas(id) ON DELETE SET NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      last_message_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
      id uuid PRIMARY KEY,
      chat_id uuid NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
      role message_role NOT NULL,
      content text NOT NULL,
      token_count int,
      meta jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_summaries (
      id uuid PRIMARY KEY,
      chat_id uuid NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
      summary_text text NOT NULL,
      from_message_id uuid REFERENCES messages(id) ON DELETE SET NULL,
      to_message_id uuid REFERENCES messages(id) ON DELETE SET NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_items (
      id uuid PRIMARY KEY,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      chat_id uuid REFERENCES chats(id) ON DELETE SET NULL,
      type memory_type NOT NULL,
      content text NOT NULL,
      importance smallint NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
      source_message_id uuid REFERENCES messages(id) ON DELETE SET NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      expires_at timestamptz
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_links (
      memory_item_id uuid NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
      message_id uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
      relevance_score numeric(4,3) NOT NULL DEFAULT 0.500
        CHECK (relevance_score >= 0.000 AND relevance_score <= 1.000),
      PRIMARY KEY(memory_item_id, message_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS todo_items (
      id uuid PRIMARY KEY,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title varchar(200) NOT NULL,
      description text,
      status varchar(20) NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'completed', 'cancelled', 'archived')),
      priority smallint NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
      due_at timestamptz,
      remind_at timestamptz,
      timezone varchar(64),
      calendar_provider varchar(80),
      calendar_id varchar(200),
      calendar_event_id varchar(200),
      calendar_sync_status varchar(20) NOT NULL DEFAULT 'none'
        CHECK (calendar_sync_status IN ('none', 'linked', 'pending', 'synced', 'failed')),
      chat_id uuid REFERENCES chats(id) ON DELETE SET NULL,
      source_message_id uuid REFERENCES messages(id) ON DELETE SET NULL,
      metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      completed_at timestamptz,
      deleted_at timestamptz
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_model_configs (
      id uuid PRIMARY KEY,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      provider_mode ai_provider_mode NOT NULL DEFAULT 'local',
      provider_name varchar(60) NOT NULL,
      model_name varchar(120) NOT NULL,
      api_key text,
      endpoint text,
      is_active boolean NOT NULL DEFAULT true,
      is_default boolean NOT NULL DEFAULT false,
      supports_stream boolean NOT NULL DEFAULT true,
      supports_realtime boolean NOT NULL DEFAULT false,
      transport varchar(30) NOT NULL DEFAULT 'http_sse',
      input_modalities varchar(120) NOT NULL DEFAULT 'text',
      output_modalities varchar(120) NOT NULL DEFAULT 'text',
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS user_ai_model_selection (
      user_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      realtime_model_config_id uuid REFERENCES ai_model_configs(id) ON DELETE SET NULL,
      deep_model_config_id uuid REFERENCES ai_model_configs(id) ON DELETE SET NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_personas_owner_name ON personas(owner_user_id, name);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_personas_user_persona ON user_personas(user_id, persona_id);",
    "CREATE INDEX IF NOT EXISTS idx_chats_user_last_message_at ON chats(user_id, last_message_at);",
    "CREATE INDEX IF NOT EXISTS idx_messages_chat_created_at ON messages(chat_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_chat_summaries_chat_created_at ON chat_summaries(chat_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_memory_items_user_importance_created ON memory_items(user_id, importance, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_memory_items_chat_created ON memory_items(chat_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_message_id ON memory_links(message_id);",
    "CREATE INDEX IF NOT EXISTS idx_todo_items_user_status_due ON todo_items(user_id, status, due_at);",
    "CREATE INDEX IF NOT EXISTS idx_todo_items_user_updated ON todo_items(user_id, updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_todo_items_calendar_ref ON todo_items(calendar_provider, calendar_id, calendar_event_id);",
    "CREATE INDEX IF NOT EXISTS idx_ai_model_configs_user_default_updated ON ai_model_configs(user_id, is_default, updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_user_ai_model_selection_realtime ON user_ai_model_selection(realtime_model_config_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_ai_model_selection_deep ON user_ai_model_selection(deep_model_config_id);",
]

SQLITE_SCRIPT = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT,
    name TEXT NOT NULL,
    description TEXT,
    prompt_template TEXT NOT NULL,
    tone TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(owner_user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_personas (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    alias TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(persona_id) REFERENCES personas(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    last_selected_user_persona_id TEXT,
    created_at TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(last_selected_user_persona_id) REFERENCES user_personas(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chat_id) REFERENCES chats(id)
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT PRIMARY KEY,
    locale TEXT NOT NULL DEFAULT 'ko-KR',
    timezone TEXT NOT NULL DEFAULT 'Asia/Seoul',
    response_style TEXT NOT NULL DEFAULT 'balanced',
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_summaries (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    from_message_id TEXT,
    to_message_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    FOREIGN KEY(from_message_id) REFERENCES messages(id) ON DELETE SET NULL,
    FOREIGN KEY(to_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    chat_id TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,
    source_message_id TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE SET NULL,
    FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_links (
    memory_item_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    relevance_score REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY(memory_item_id, message_id),
    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS todo_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'completed', 'cancelled', 'archived')),
    priority INTEGER NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    due_at TEXT,
    remind_at TEXT,
    timezone TEXT,
    calendar_provider TEXT,
    calendar_id TEXT,
    calendar_event_id TEXT,
    calendar_sync_status TEXT NOT NULL DEFAULT 'none' CHECK (calendar_sync_status IN ('none', 'linked', 'pending', 'synced', 'failed')),
    chat_id TEXT,
    source_message_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    deleted_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE SET NULL,
    FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ai_model_configs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider_mode TEXT NOT NULL DEFAULT 'local',
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    endpoint TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    is_default INTEGER NOT NULL DEFAULT 0,
    supports_stream INTEGER NOT NULL DEFAULT 1,
    supports_realtime INTEGER NOT NULL DEFAULT 0,
    transport TEXT NOT NULL DEFAULT 'http_sse',
    input_modalities TEXT NOT NULL DEFAULT 'text',
    output_modalities TEXT NOT NULL DEFAULT 'text',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_ai_model_selection (
    user_id TEXT PRIMARY KEY,
    realtime_model_config_id TEXT,
    deep_model_config_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(realtime_model_config_id) REFERENCES ai_model_configs(id) ON DELETE SET NULL,
    FOREIGN KEY(deep_model_config_id) REFERENCES ai_model_configs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_personas_owner_name ON personas(owner_user_id, name);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_personas_user_persona ON user_personas(user_id, persona_id);
CREATE INDEX IF NOT EXISTS idx_chats_user_last_message_at ON chats(user_id, last_message_at);
CREATE INDEX IF NOT EXISTS idx_messages_chat_created_at ON messages(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_summaries_chat_created_at ON chat_summaries(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_items_user_importance_created ON memory_items(user_id, importance, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_items_chat_created ON memory_items(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_links_message_id ON memory_links(message_id);
CREATE INDEX IF NOT EXISTS idx_todo_items_user_status_due ON todo_items(user_id, status, due_at);
CREATE INDEX IF NOT EXISTS idx_todo_items_user_updated ON todo_items(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_todo_items_calendar_ref ON todo_items(calendar_provider, calendar_id, calendar_event_id);
CREATE INDEX IF NOT EXISTS idx_ai_model_configs_user_default_updated ON ai_model_configs(user_id, is_default, updated_at);
CREATE INDEX IF NOT EXISTS idx_user_ai_model_selection_realtime ON user_ai_model_selection(realtime_model_config_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_model_selection_deep ON user_ai_model_selection(deep_model_config_id);
"""


def _migrate_postgres_ai_model_configs(db: DBClient) -> None:
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS is_default boolean
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS supports_stream boolean
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS supports_realtime boolean
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS transport varchar(30)
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS input_modalities varchar(120)
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ADD COLUMN IF NOT EXISTS output_modalities varchar(120)
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN is_default SET DEFAULT false
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET is_default = false
        WHERE is_default IS NULL
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET supports_stream = true
        WHERE supports_stream IS NULL
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET supports_realtime = false
        WHERE supports_realtime IS NULL
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET transport = 'http_sse'
        WHERE transport IS NULL
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET input_modalities = 'text'
        WHERE input_modalities IS NULL
        """
    )
    db.conn.execute(
        """
        UPDATE ai_model_configs
        SET output_modalities = 'text'
        WHERE output_modalities IS NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN is_default SET NOT NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN supports_stream SET DEFAULT true
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN supports_realtime SET DEFAULT false
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN transport SET DEFAULT 'http_sse'
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN input_modalities SET DEFAULT 'text'
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN output_modalities SET DEFAULT 'text'
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN supports_stream SET NOT NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN supports_realtime SET NOT NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN transport SET NOT NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN input_modalities SET NOT NULL
        """
    )
    db.conn.execute(
        """
        ALTER TABLE IF EXISTS ai_model_configs
        ALTER COLUMN output_modalities SET NOT NULL
        """
    )
    db.conn.execute(
        """
        DO $$
        DECLARE unique_constraint_name text;
        BEGIN
            IF to_regclass('ai_model_configs') IS NULL THEN
                RETURN;
            END IF;

            SELECT c.conname INTO unique_constraint_name
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE t.relname = 'ai_model_configs'
              AND n.nspname = current_schema()
              AND c.contype = 'u'
              AND array_length(c.conkey, 1) = 1
              AND c.conkey[1] = (
                  SELECT a.attnum
                  FROM pg_attribute a
                  WHERE a.attrelid = t.oid
                    AND a.attname = 'user_id'
                    AND a.attisdropped = false
                  LIMIT 1
              )
            LIMIT 1;

            IF unique_constraint_name IS NOT NULL THEN
                EXECUTE format('ALTER TABLE ai_model_configs DROP CONSTRAINT %I', unique_constraint_name);
            END IF;
        END$$;
        """
    )
    db.conn.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY is_default DESC, updated_at DESC, created_at DESC, id DESC
                ) AS rn
            FROM ai_model_configs
            WHERE is_active = true
        )
        UPDATE ai_model_configs AS t
        SET is_default = CASE WHEN r.rn = 1 THEN true ELSE false END
        FROM ranked AS r
        WHERE t.id = r.id
        """
    )


def _migrate_sqlite_ai_model_configs(db: DBClient) -> None:
    table_row = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'ai_model_configs'"
    ).fetchone()
    if table_row is None:
        return

    create_sql = (table_row[0] or "").upper()
    table_info = db.conn.execute("PRAGMA table_info(ai_model_configs)").fetchall()
    has_is_default = any(col[1] == "is_default" for col in table_info)
    has_supports_stream = any(col[1] == "supports_stream" for col in table_info)
    has_supports_realtime = any(col[1] == "supports_realtime" for col in table_info)
    has_transport = any(col[1] == "transport" for col in table_info)
    has_input_modalities = any(col[1] == "input_modalities" for col in table_info)
    has_output_modalities = any(col[1] == "output_modalities" for col in table_info)
    has_user_unique = "USER_ID" in create_sql and "UNIQUE" in create_sql

    if (
        has_is_default
        and has_supports_stream
        and has_supports_realtime
        and has_transport
        and has_input_modalities
        and has_output_modalities
        and not has_user_unique
    ):
        db.conn.execute(
            """
            WITH ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY is_default DESC, updated_at DESC, created_at DESC, id DESC
                    ) AS rn
                FROM ai_model_configs
                WHERE is_active = 1
            )
            UPDATE ai_model_configs
            SET is_default = CASE
                WHEN id IN (SELECT id FROM ranked WHERE rn = 1) THEN 1
                ELSE 0
            END
            WHERE id IN (SELECT id FROM ranked)
            """
        )
        return

    db.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_model_configs_new (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            provider_mode TEXT NOT NULL DEFAULT 'local',
            provider_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            api_key TEXT,
            endpoint TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,
            supports_stream INTEGER NOT NULL DEFAULT 1,
            supports_realtime INTEGER NOT NULL DEFAULT 0,
            transport TEXT NOT NULL DEFAULT 'http_sse',
            input_modalities TEXT NOT NULL DEFAULT 'text',
            output_modalities TEXT NOT NULL DEFAULT 'text',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    select_is_default = "is_default" if has_is_default else "0"
    select_supports_stream = "supports_stream" if has_supports_stream else "1"
    select_supports_realtime = "supports_realtime" if has_supports_realtime else "0"
    select_transport = "transport" if has_transport else "'http_sse'"
    select_input_modalities = "input_modalities" if has_input_modalities else "'text'"
    select_output_modalities = "output_modalities" if has_output_modalities else "'text'"

    db.conn.execute(
        f"""
        INSERT INTO ai_model_configs_new (
            id, user_id, provider_mode, provider_name, model_name, api_key, endpoint, is_active, is_default,
            supports_stream, supports_realtime, transport, input_modalities, output_modalities, created_at, updated_at
        )
        SELECT
            id, user_id, provider_mode, provider_name, model_name, api_key, endpoint, is_active,
            {select_is_default},
            {select_supports_stream},
            {select_supports_realtime},
            {select_transport},
            {select_input_modalities},
            {select_output_modalities},
            created_at, updated_at
        FROM ai_model_configs
        """
    )

    db.conn.execute("DROP TABLE ai_model_configs")
    db.conn.execute("ALTER TABLE ai_model_configs_new RENAME TO ai_model_configs")
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_model_configs_user_default_updated ON ai_model_configs(user_id, is_default, updated_at)"
    )
    db.conn.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY is_default DESC, updated_at DESC, created_at DESC, id DESC
                ) AS rn
            FROM ai_model_configs
            WHERE is_active = 1
        )
        UPDATE ai_model_configs
        SET is_default = CASE
            WHEN id IN (SELECT id FROM ranked WHERE rn = 1) THEN 1
            ELSE 0
        END
        WHERE id IN (SELECT id FROM ranked)
        """
    )


def init_db(db: DBClient) -> None:
    if db.backend == "postgres":
        # Serialize schema initialization across services to avoid deadlocks
        # when core and gateway start at the same time against the same DB.
        db.conn.execute("SELECT pg_advisory_xact_lock(%s)", (684174205901234567,))
        for statement in POSTGRES_STATEMENTS:
            db.conn.execute(statement)
        _migrate_postgres_ai_model_configs(db)
        db.conn.commit()
        return

    db.conn.executescript(SQLITE_SCRIPT)
    _migrate_sqlite_ai_model_configs(db)
    db.conn.commit()
    print("Database initialized")
