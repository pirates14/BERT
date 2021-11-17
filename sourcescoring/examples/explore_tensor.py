import torch
import numpy as np

def main():
    # 텐서를 사용하는 예제

    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    # np배열로 텐서 생성하기
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(x_np)

    # ones_like -> 아 이거 다 1로 깔아버리네
    x_ones = torch.ones_like(x_data)
    print(x_ones)

    # rand_like -> 랜덤 숫자로 깔아버리는데
    # x_rand = torch.rand_like(x_data) 이렇게만 하니까
    # RuntimeError: "check_uniform_bounds" not implemented for 'Long'
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    # torch.float를 꼭 데이터타입으로 넣어줘야하나?
    # 일단 torch.int이건 안됨 check_uniform_bounds" not implemented for 'Int'
    # 몇 번을 돌려도 1보다 더 큰 실수는 안나옴
    print(x_rand)

    hape = (2, 3,)
    # tensor dim을 2,3으로 맞춰만 주고 아래에 .rand .ones .zeros 활용해서 값을 넣는거?
    rand_tensor = torch.rand(hape)  # 엇 여기서도 rand인데 실수로 나오긴 했네
    one_tensor = torch.ones(hape)
    zero_tensor = torch.zeros(hape)

    print(rand_tensor)
    print(one_tensor)
    print(zero_tensor)

    # 굳이 변수로 dim 정해줄 필요 없이 바로 넣는것도 오케이데스
    tenso = torch.rand(3, 4)
    print(tenso)
    print(tenso.shape)
    print(tenso.dtype)  # 왜 float32지 rand은 default가 실수인가? ㅇㅇ그런거 같다
    print(tenso.device)  # 얜 뭐지 CPU.. Device tensor is stored on라는데
    # .device가 있을 정도로 텐서가 어디 저장되어있는지가 중요한건가?


if __name__ == '__main__':
    main()
