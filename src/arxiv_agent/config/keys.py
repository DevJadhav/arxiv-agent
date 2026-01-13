"""Secure API key storage with encryption."""

import base64
import json
import os
from pathlib import Path
from typing import Literal

from loguru import logger

# Try to use keyring first, fall back to encrypted file storage
try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

# For file-based encryption fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


SERVICE_NAME = "arxiv-agent"
PROVIDERS = ("anthropic", "openai", "gemini", "semantic_scholar")

ProviderType = Literal["anthropic", "openai", "gemini", "semantic_scholar"]


class KeyStorage:
    """Secure API key storage manager.
    
    Uses system keyring when available, falls back to encrypted file storage.
    """
    
    def __init__(self, config_dir: Path | None = None):
        """Initialize key storage.
        
        Args:
            config_dir: Configuration directory for encrypted file fallback
        """
        if config_dir:
            self.config_dir = config_dir
        else:
            from arxiv_agent.config.settings import get_settings
            self.config_dir = get_settings().config_dir
        
        self.keys_file = self.config_dir / ".keys.enc"
        self._encryption_key: bytes | None = None
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for file-based storage."""
        if self._encryption_key:
            return self._encryption_key
        
        key_file = self.config_dir / ".keyfile"
        
        if key_file.exists():
            self._encryption_key = key_file.read_bytes()
        else:
            # Generate new key
            if HAS_CRYPTO:
                # Use machine-specific salt
                salt = os.urandom(16)
                machine_id = (os.getlogin() + str(Path.home())).encode()
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=480000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(machine_id))
                
                # Store key (secured by file permissions)
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_bytes(salt + key)
                os.chmod(key_file, 0o600)
                
                self._encryption_key = key
            else:
                # Fallback: use base64 (not secure, but better than plaintext)
                self._encryption_key = base64.urlsafe_b64encode(b"arxiv-agent-default-key-32b!")
        
        return self._encryption_key
    
    def _load_file_keys(self) -> dict[str, str]:
        """Load keys from encrypted file."""
        if not self.keys_file.exists():
            return {}
        
        try:
            if HAS_CRYPTO:
                key = self._get_encryption_key()
                f = Fernet(key)
                encrypted = self.keys_file.read_bytes()
                decrypted = f.decrypt(encrypted)
                return json.loads(decrypted.decode())
            else:
                # Fallback: base64 only
                encoded = self.keys_file.read_bytes()
                decoded = base64.b64decode(encoded)
                return json.loads(decoded.decode())
        except Exception as e:
            logger.warning(f"Failed to load keys file: {e}")
            return {}
    
    def _save_file_keys(self, keys: dict[str, str]) -> None:
        """Save keys to encrypted file."""
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = json.dumps(keys).encode()
            
            if HAS_CRYPTO:
                key = self._get_encryption_key()
                f = Fernet(key)
                encrypted = f.encrypt(data)
                self.keys_file.write_bytes(encrypted)
            else:
                # Fallback: base64 only
                encoded = base64.b64encode(data)
                self.keys_file.write_bytes(encoded)
            
            os.chmod(self.keys_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
            raise
    
    def set_key(self, provider: ProviderType, api_key: str) -> bool:
        """Store an API key securely.
        
        Args:
            provider: Provider name (anthropic, openai, gemini, semantic_scholar)
            api_key: The API key to store
        
        Returns:
            True if successful
        """
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Must be one of {PROVIDERS}")
        
        # Try keyring first
        if HAS_KEYRING:
            try:
                keyring.set_password(SERVICE_NAME, provider, api_key)
                logger.debug(f"Stored {provider} key in system keyring")
                return True
            except Exception as e:
                logger.debug(f"Keyring storage failed, using file: {e}")
        
        # Fall back to encrypted file
        keys = self._load_file_keys()
        keys[provider] = api_key
        self._save_file_keys(keys)
        logger.debug(f"Stored {provider} key in encrypted file")
        return True
    
    def get_key(self, provider: ProviderType) -> str | None:
        """Retrieve an API key.
        
        Args:
            provider: Provider name
        
        Returns:
            API key or None if not found
        """
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Check environment variable first
        env_var = f"ARXIV_AGENT_{provider.upper()}_API_KEY"
        if env_key := os.environ.get(env_var):
            return env_key
        
        # Try keyring
        if HAS_KEYRING:
            try:
                if key := keyring.get_password(SERVICE_NAME, provider):
                    return key
            except Exception:
                pass
        
        # Try encrypted file
        keys = self._load_file_keys()
        return keys.get(provider)
    
    def delete_key(self, provider: ProviderType) -> bool:
        """Delete an API key.
        
        Args:
            provider: Provider name
        
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        # Try keyring
        if HAS_KEYRING:
            try:
                keyring.delete_password(SERVICE_NAME, provider)
                deleted = True
            except Exception:
                pass
        
        # Try file
        keys = self._load_file_keys()
        if provider in keys:
            del keys[provider]
            self._save_file_keys(keys)
            deleted = True
        
        return deleted
    
    def list_configured(self) -> dict[str, bool]:
        """List which providers have keys configured.
        
        Returns:
            Dict mapping provider name to configured status
        """
        result = {}
        for provider in PROVIDERS:
            result[provider] = self.get_key(provider) is not None
        return result
    
    def get_masked_key(self, provider: ProviderType) -> str | None:
        """Get a masked version of the key for display.
        
        Args:
            provider: Provider name
        
        Returns:
            Masked key (e.g., "sk-ant-***...***") or None
        """
        key = self.get_key(provider)
        if not key:
            return None
        
        if len(key) <= 12:
            return "*" * len(key)
        
        return f"{key[:8]}...{key[-4:]}"


# Global instance
_key_storage: KeyStorage | None = None


def get_key_storage() -> KeyStorage:
    """Get key storage instance."""
    global _key_storage
    if _key_storage is None:
        _key_storage = KeyStorage()
    return _key_storage
