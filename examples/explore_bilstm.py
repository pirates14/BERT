
import torch


def main():
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    N, L, H = (10, 30, 768)
    H_all = torch.randn(size=(N, L, H))
    assert H % 2 == 0
    bilstm = torch.nn.LSTM(input_size=H,  # 입력으로 들어오는 행렬의 차원
                           # 2개의 lstm을 concat 하므로, 기존의 hidden_size 를 유지하기 위해 절반으로 설정
                           bidirectional=True, hidden_size=H // 2,
                           #  (N, ...) 이 입력이다는 걸 알리기
                           batch_first=True
    )
    # (N, L, H) -> (N, L, H), (N, L, H // 2), (N, L, H // 2)
    hidden_states, (cell_states_lr, cell_states_rl) = bilstm(H_all)
    print(hidden_states.shape)
    print(cell_states_lr.shape)
    print(cell_states_rl.shape)

    # 출력은 다음과 같이 나올 것
    """
    torch.Size([10, 30, 768])
    torch.Size([2, 10, 384])
    torch.Size([2, 10, 384])
    """
    # lstm을 통과한 히든벡터로 (즉 순서의 정보가 더 담겨있는) 기존의 히든벡터를 대체
    H_all = hidden_states


if __name__ == '__main__':
    main()
