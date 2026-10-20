def t1_grid(T: int, N: int, n1: int, n2: int):
    if not (0 <= n1 <= n2 <= N):
        raise ValueError(f'need 0 <= n1 <= n2 <= N, got n1={n1} n2={n2} N={N}')
    return [round(n * T / N) for n in range(n1, n2 + 1)]
