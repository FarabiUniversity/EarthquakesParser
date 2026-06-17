"""Supabase file storage utility - low-level file operations."""

import os
from pathlib import Path
from typing import Optional


def _load_local_env() -> None:
    """Load Supabase env vars from common repo-local `.env` locations."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[3]
    for env_path in (repo_root / ".env", repo_root / "veritatis" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_local_env()


class SupabaseFileStorage:
    """Low-level Supabase Storage (S3-compatible) file operations.

    This is a utility class providing generic file upload/download.
    Business logic should be in domain modules (parser, searcher).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: str = "storage",
    ):
        """Initialize Supabase file storage client.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase service role key. Defaults to SUPABASE_KEY env var.
            bucket_name: Name of the storage bucket.

        Raises:
            ImportError: If supabase package is not installed.
            ValueError: If URL or key is missing.
        """
        try:
            from supabase import create_client  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "supabase package is required. Install with: pip install supabase"
            ) from e

        # Get credentials
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key are required. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables "
                "or pass them explicitly."
            )

        # Initialize client
        self.client = create_client(self.url, self.key)
        self.bucket_name = bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create storage bucket if it doesn't exist."""
        try:
            buckets = self.client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]

            if self.bucket_name not in bucket_names:
                self.client.storage.create_bucket(
                    self.bucket_name, options={"public": False}
                )
        except Exception as e:
            print(f"Warning: Could not verify/create bucket: {e}")

    def upload(
        self, path: str, content: str, content_type: str = "text/plain"
    ) -> Optional[str]:
        """Upload file content to storage.

        Args:
            path: Full path in bucket (e.g., "html/file123.html").
            content: File content as string.
            content_type: MIME type (default: text/plain).

        Returns:
            Storage path if successful, None otherwise.
        """
        try:
            content_bytes = content.encode("utf-8")
            self.client.storage.from_(self.bucket_name).upload(
                path, content_bytes, {"content-type": content_type}
            )
            return path

        except Exception as e:
            print(f"Error uploading file to storage: {e}")
            return None

    def download(self, path: str) -> Optional[str]:
        """Download file content from storage.

        Args:
            path: Path to the file in storage.

        Returns:
            File content as string, or None if not found.
        """
        try:
            response = self.client.storage.from_(self.bucket_name).download(path)
            return str(response.decode("utf-8"))

        except Exception as e:
            print(f"Error downloading file from storage: {e}")
            return None

    def delete(self, path: str) -> bool:
        """Delete file from storage.

        Args:
            path: Path to the file in storage.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.storage.from_(self.bucket_name).remove([path])
            return True

        except Exception as e:
            print(f"Error deleting file from storage: {e}")
            return False

    def exists(self, path: str) -> bool:
        """Check if file exists in storage.

        Args:
            path: Path to the file in storage.

        Returns:
            True if file exists, False otherwise.
        """
        try:
            self.client.storage.from_(self.bucket_name).download(path)
            return True

        except Exception:
            return False

    def list_files(self, folder: str = "") -> list:
        """List files in a folder.

        Args:
            folder: Folder path (empty string for root).

        Returns:
            List of file objects.
        """
        try:
            response = self.client.storage.from_(self.bucket_name).list(folder)
            return list(response)

        except Exception as e:
            print(f"Error listing files: {e}")
            return []
