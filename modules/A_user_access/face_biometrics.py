"""
Face capture / verification — FaceNet deep embeddings (facenet-pytorch).
=======================================================================
Uses MTCNN for face detection + alignment, and InceptionResnetV1 (VGGFace2)
for 512-dimensional face embeddings that actually distinguish between
different people.

Previous approach (Haar cascade + raw pixel cosine) was broken:
any two face crops scored > 0.82 because all faces look structurally similar
at the pixel level.  Deep embeddings capture *identity*, not pixel patterns.
"""

from __future__ import annotations

import numpy as np

from config import FACE_COSINE_THRESHOLD

# Lazy-loaded models — first call downloads ~100 MB of weights
_mtcnn = None
_resnet = None


def _get_models():
    """Lazy-load MTCNN detector + InceptionResnetV1 encoder."""
    global _mtcnn, _resnet
    if _mtcnn is None:
        print("[DEBUG face] Loading MTCNN + InceptionResnetV1 (first call downloads weights)…")
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch

        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,        # return only the largest face
            post_process=True,     # normalize to [-1, 1]
            device="cpu",
        )
        _resnet = InceptionResnetV1(pretrained="vggface2").eval()
        print("[DEBUG face] Models loaded successfully.")
    return _mtcnn, _resnet


def extract_face_encoding(rgb_image: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest face and return a 512-dim FaceNet embedding.

    Returns None if no face is detected.
    """
    from PIL import Image
    import torch

    print(f"[DEBUG face] extract_face_encoding: input shape={rgb_image.shape}, "
          f"dtype={rgb_image.dtype}")

    mtcnn, resnet = _get_models()

    # MTCNN expects a PIL Image
    pil_img = Image.fromarray(rgb_image)
    face_tensor = mtcnn(pil_img)          # detect → crop → align → tensor

    if face_tensor is None:
        print("[DEBUG face] extract_face_encoding: NO face detected!")
        return None

    print(f"[DEBUG face] extract_face_encoding: face tensor shape={face_tensor.shape}")

    # Get 512-dim embedding
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0))

    result = embedding.squeeze().cpu().numpy().astype(np.float32)
    print(f"[DEBUG face] extract_face_encoding: embedding shape={result.shape}, "
          f"norm={np.linalg.norm(result):.4f}")
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors — 1.0 = identical, 0.0 = orthogonal."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        print(f"[DEBUG face] cosine_similarity: ZERO norm! na={na:.6f}, nb={nb:.6f}")
        return 0.0
    score = float(np.dot(a, b) / (na * nb))
    print(f"[DEBUG face] cosine_similarity: score={score:.6f}")
    return score


def compare_faces(template: np.ndarray, probe: np.ndarray) -> tuple[float, bool]:
    """
    Compare two FaceNet embeddings.

    Returns (cosine_score, is_match).
    Same person → score typically 0.6–0.9
    Different person → score typically 0.0–0.4
    """
    print(f"[DEBUG face] compare_faces: template shape={template.shape}, "
          f"probe shape={probe.shape}")
    score = cosine_similarity(template, probe)
    is_match = score >= FACE_COSINE_THRESHOLD
    print(f"[DEBUG face] compare_faces: score={score:.6f} vs "
          f"threshold={FACE_COSINE_THRESHOLD} → MATCH={is_match}")
    return score, is_match
