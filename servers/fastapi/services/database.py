from collections.abc import AsyncGenerator
import os
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel

from models.sql.async_presentation_generation_status import (
    AsyncPresentationGenerationTaskModel,
)
from models.sql.image_asset import ImageAsset
from models.sql.key_value import KeyValueSqlModel
from models.sql.ollama_pull_status import OllamaPullStatus
from models.sql.presentation import PresentationModel
from models.sql.slide import SlideModel
from models.sql.presentation_layout_code import PresentationLayoutCodeModel
from models.sql.template import TemplateModel
from models.sql.webhook_subscription import WebhookSubscription
from utils.db_utils import get_database_url_and_connect_args


# ---------------------------------------------------------------------------
# ✅ DATABASE PATHS (Render-safe)
# ---------------------------------------------------------------------------
os.makedirs("/tmp/app_data", exist_ok=True)

# Main database in /tmp (Render’s writable folder)
MAIN_DB_PATH = "/tmp/app_data/presenton.db"
CONTAINER_DB_PATH = "/tmp/app_data/container.db"

database_url, connect_args = get_database_url_and_connect_args()

# If your get_database_url_and_connect_args() uses env vars, override for safety:
if not database_url.startswith("sqlite"):
    database_url = f"sqlite+aiosqlite:///{MAIN_DB_PATH}"

sql_engine: AsyncEngine = create_async_engine(database_url, connect_args=connect_args)
async_session_maker = async_sessionmaker(sql_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


# ---------------------------------------------------------------------------
# ✅ CONTAINER DB SETUP (Ollama status, etc.)
# ---------------------------------------------------------------------------
container_db_url = f"sqlite+aiosqlite:///{CONTAINER_DB_PATH}"
container_db_engine: AsyncEngine = create_async_engine(
    container_db_url, connect_args={"check_same_thread": False}
)
container_db_async_session_maker = async_sessionmaker(
    container_db_engine, expire_on_commit=False
)


async def get_container_db_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with container_db_async_session_maker() as session:
        yield session


# ---------------------------------------------------------------------------
# ✅ CREATE DATABASE & TABLES (Safe for Render)
# ---------------------------------------------------------------------------
async def create_db_and_tables():
    # 🧹 Auto-clean broken DB if necessary
    if os.path.exists(MAIN_DB_PATH):
        try:
            with open(MAIN_DB_PATH, "rb") as f:
                if b"ix_slides_presentation" in f.read():
                    print("⚠️ Removing old database to fix duplicate index error.")
                    os.remove(MAIN_DB_PATH)
        except Exception as e:
            print(f"⚠️ Skipping DB cleanup due to: {e}")

    # ✅ MAIN DATABASE
    async with sql_engine.begin() as conn:
        try:
            await conn.run_sync(
                lambda sync_conn: SQLModel.metadata.create_all(
                    sync_conn,
                    tables=[
                        PresentationModel.__table__,
                        SlideModel.__table__,
                        KeyValueSqlModel.__table__,
                        ImageAsset.__table__,
                        PresentationLayoutCodeModel.__table__,
                        TemplateModel.__table__,
                        WebhookSubscription.__table__,
                        AsyncPresentationGenerationTaskModel.__table__,
                    ],
                )
            )
        except OperationalError as e:
            if "already exists" in str(e):
                print("⚠️ Duplicate index detected — skipping creation.")
            else:
                raise

    # ✅ CONTAINER DATABASE
    async with container_db_engine.begin() as conn:
        try:
            await conn.run_sync(
                lambda sync_conn: SQLModel.metadata.create_all(
                    sync_conn, tables=[OllamaPullStatus.__table__]
                )
            )
        except OperationalError as e:
            if "already exists" in str(e):
                print("⚠️ Skipping duplicate index creation in container DB.")
            else:
                raise
