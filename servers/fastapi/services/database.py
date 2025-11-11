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
# ✅ DATABASE ENGINES SETUP
# ---------------------------------------------------------------------------

database_url, connect_args = get_database_url_and_connect_args()

sql_engine: AsyncEngine = create_async_engine(database_url, connect_args=connect_args)
async_session_maker = async_sessionmaker(sql_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


# ---------------------------------------------------------------------------
# ✅ CONTAINER DB SETUP (used for Ollama model pull statuses, etc.)
# ---------------------------------------------------------------------------
container_db_url = "sqlite+aiosqlite:////app/container.db"
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
# ✅ CREATE DATABASE & TABLES (with Render-safe auto cleanup)
# ---------------------------------------------------------------------------
async def create_db_and_tables():
    db_path = "/opt/render/project/src/app_data/presenton.db"

    # 🧹 Detect and remove corrupted DB file causing "ix_slides_presentation already exists"
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                if b"ix_slides_presentation" in f.read():
                    print("⚠️ Removing old database to fix duplicate index error.")
                    os.remove(db_path)
        except Exception as e:
            print(f"⚠️ Skipping DB cleanup due to: {e}")

    # ✅ MAIN DATABASE INITIALIZATION
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

    # ✅ CONTAINER DATABASE INITIALIZATION
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
