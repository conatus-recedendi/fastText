# pip install fasttext
import fasttext
import numpy as np
import matplotlib.pyplot as plt


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A @ B.T


def get_subword_vectors(
    model,
    word: str,
    nmin: int,
    nmax: int,
    include_word: bool = True,
):
    """
    fastText 모델에서 word의 subword 문자열과 그 벡터를 가져옵니다.
    - nmin/nmax가 None이면 모델 학습 시 사용된 minn/maxn을 그대로 따릅니다.
    - include_word=True이면 사전에 있는 단어 자체 벡터도 포함(중복 id 제거).
    반환: (subword_list, vectors[np.ndarray, shape=(S, dim)])
    """
    # 기본값: 모델 설정 값 사용
    args = model.get_args()
    if nmin is None:
        nmin = args.minn
    if nmax is None:
        nmax = args.maxn

    subs, idxs = model.get_subwords(word)  # subword 문자열, 해당 input-matrix 인덱스
    pairs = []
    for s, i in zip(subs, idxs):
        if i < 0:
            continue
        # 길이 필터(경계 기호 '<', '>' 포함 길이 기준)
        if len(s) < nmin or len(s) > nmax:
            continue
        v = np.array(model.get_input_vector(i))
        pairs.append((s, i, v))

    # 단어 자체 벡터 포함(사전에 있는 경우)
    if include_word:
        wid = model.get_word_id(word)
        if wid != -1:
            v = np.array(model.get_input_vector(wid))
            pairs.insert(0, (word, wid, v))

    # 같은 인덱스 중복 제거(해시 충돌/중복 방지)
    seen, uniq = set(), []
    for s, i, v in pairs:
        if i in seen:
            continue
        seen.add(i)
        uniq.append((s, v))

    labels = [s for s, _ in uniq]
    vecs = (
        np.vstack([v for _, v in uniq])
        if uniq
        else np.zeros((0, model.get_dimension()))
    )
    return labels, vecs


def plot_subword_heatmap(
    model_path: str,
    w1: str,
    w2: str,
    nmin: int,
    nmax: int,
    include_word: bool = True,
    save_path: str = None,
):
    """
    두 단어의 subword 간 코사인 유사도 히트맵을 그립니다.
    """
    model = fasttext.load_model(model_path)

    s1, V1 = get_subword_vectors(model, w1, nmin, nmax, include_word=include_word)
    s2, V2 = get_subword_vectors(model, w2, nmin, nmax, include_word=include_word)

    if V1.shape[0] == 0 or V2.shape[0] == 0:
        raise ValueError(
            "선택한 n-gram 범위에 해당하는 subword가 없습니다. nmin/nmax를 확인하세요."
        )

    M = _cosine_sim_matrix(V1, V2)

    plt.figure(figsize=(8, 6))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.xticks(ticks=np.arange(len(s2)), labels=s2, rotation=70, ha="right", fontsize=8)
    plt.yticks(ticks=np.arange(len(s1)), labels=s1, fontsize=8)
    plt.xlabel(w2)
    plt.ylabel(w1)
    plt.title(f"fastText subword cosine similarity: {w1} vs {w2}  [n={nmin}..{nmax}]")
    plt.colorbar(label="cosine similarity")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return s1, s2, M


# ---------- 사용 예시 ----------
model_path = "./result/re-ft-sisg-wiki-en.bin"  # 본인 모델 경로
plot_subword_heatmap(
    model_path,
    "rarity",
    "scarceness",
    nmin=3,
    nmax=4,
    include_word=True,
    save_path="p2_figure2_rarity_scarceness.png",
)
