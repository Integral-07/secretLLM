import os
import hmac
import hashlib
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives import hashes


class SessionKeySet:
	"""1セッション分の派生鍵セット。"""

	def __init__(self, session_secret: bytes):
		self._session_secret = session_secret

	def derive_component_bytes(self, layer_idx: int, component: str, num_bytes: int) -> bytes:
		info = f"layer:{layer_idx}:{component}".encode()
		if num_bytes <= 8160:
			return HKDFExpand(
				algorithm=hashes.SHA256(),
				length=num_bytes,
				info=info,
			).derive(self._session_secret)
		else:
			return self._derive_large(info, num_bytes)

	def _derive_large(self, base_info: bytes, num_bytes: int) -> bytes:
		chunks = []
		remaining = num_bytes
		chunk_idx = 0
		while remaining > 0:
			chunk_size = min(remaining, 8160)
			chunk_info = base_info + f":chunk:{chunk_idx}".encode()
			chunk = HKDFExpand(
				algorithm=hashes.SHA256(),
				length=chunk_size,
				info=chunk_info,
			).derive(self._session_secret)
			chunks.append(chunk)
			remaining -= chunk_size
			chunk_idx += 1
		return b"".join(chunks)


class KeyManager:
	VERSION = b"secret-llm-v1"

	def __init__(self, master_secret: bytes):
		# HKDF-Extract = HMAC-SHA256(salt, input_key_material)
		self._prk = hmac.new(
			key=self.VERSION,
			msg=master_secret,
			digestmod=hashlib.sha256,
		).digest()

	def derive_session(self, session_id: str) -> SessionKeySet:
		# HKDF-Expandのみでセッション鍵を導出
		session_secret = HKDFExpand(
			algorithm=hashes.SHA256(),
			length=32,
			info=b"session:" + session_id.encode(),
		).derive(self._prk)
		return SessionKeySet(session_secret)

	@staticmethod
	def generate_master_secret() -> bytes:
		return os.urandom(32)
