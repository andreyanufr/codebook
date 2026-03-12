import torch

def pack_2bit(x: torch.Tensor) -> torch.Tensor:
    """
    Pack a uint8 tensor with values in [0,3] into a tensor 4x smaller.

    Input shape: (..., N)  where N % 4 == 0
    Output shape: (..., N//4)
    """
    if x.dtype != torch.uint8:
        raise ValueError("Input must be uint8")
    if torch.any(x > 3):
        raise ValueError("Values must be in range [0, 3]")
    if x.shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4")

    x = x.view(*x.shape[:-1], -1, 4)

    packed = (
        (x[..., 0]      ) |
        (x[..., 1] << 2 ) |
        (x[..., 2] << 4 ) |
        (x[..., 3] << 6 )
    )

    return packed


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack 2-bit packed tensor back to uint8 values in [0,3]
    """
    if packed.dtype != torch.uint8:
        raise ValueError("Input must be uint8")

    x0 =  packed        & 0b11
    x1 = (packed >> 2 ) & 0b11
    x2 = (packed >> 4 ) & 0b11
    x3 = (packed >> 6 ) & 0b11

    return torch.stack([x0, x1, x2, x3], dim=-1).reshape(*packed.shape[:-1], -1)


def pack_3bit(x: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 tensor with values in [0,7] into tensor 3/8 the size.

    Input shape: (..., N)  where N % 8 == 0
    Output shape: (..., N*3//8)
    """
    if x.dtype != torch.uint8:
        raise ValueError("Input must be uint8")
    if torch.any(x > 7):
        raise ValueError("Values must be in range [0, 7]")
    if x.shape[-1] % 8 != 0:
        raise ValueError("Last dimension must be divisible by 8")

    batch_shape = x.shape[:-1]
    x = x.view(*batch_shape, -1, 8).to(torch.int32)

    byte0 = (x[..., 0]) | (x[..., 1] << 3) | (x[..., 2] << 6)
    byte1 = (x[..., 2] >> 2) | (x[..., 3] << 1) | (x[..., 4] << 4) | (x[..., 5] << 7)
    byte2 = (x[..., 5] >> 1) | (x[..., 6] << 2) | (x[..., 7] << 5)

    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    return packed.reshape(*batch_shape, -1).to(torch.uint8)


def unpack_3bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack 3-bit packed tensor back to uint8 values in [0,7]
    """
    if packed.dtype != torch.uint8:
        raise ValueError("Input must be uint8")
    if packed.shape[-1] % 3 != 0:
        raise ValueError("Last dimension must be divisible by 3")

    batch_shape = packed.shape[:-1]
    packed = packed.view(*batch_shape, -1, 3).to(torch.int32)

    byte0 = packed[..., 0]
    byte1 = packed[..., 1]
    byte2 = packed[..., 2]

    v0 =  byte0       & 0x07
    v1 = (byte0 >> 3) & 0x07
    v2 = ((byte0 >> 6) | (byte1 << 2)) & 0x07
    v3 = (byte1 >> 1) & 0x07
    v4 = (byte1 >> 4) & 0x07
    v5 = ((byte1 >> 7) | (byte2 << 1)) & 0x07
    v6 = (byte2 >> 2) & 0x07
    v7 = (byte2 >> 5) & 0x07

    return torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(*batch_shape, -1).to(torch.uint8)


def pack_4bit(x: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 tensor with values in [0,15] into tensor 2x smaller.

    Input shape: (..., N)  where N % 2 == 0
    Output shape: (..., N//2)
    """
    if x.dtype != torch.uint8:
        raise ValueError("Input must be uint8")
    if torch.any(x > 15):
        raise ValueError("Values must be in range [0, 15]")
    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be divisible by 2")

    x = x.view(*x.shape[:-1], -1, 2)

    packed = (x[..., 0]) | (x[..., 1] << 4)
    return packed


def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack packed 4-bit tensor back to uint8 values in [0,15]
    """
    if packed.dtype != torch.uint8:
        raise ValueError("Input must be uint8")

    low  =  packed        & 0x0F
    high = (packed >> 4 ) & 0x0F

    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


if __name__ == "__main__":
    x = torch.randint(0, 4, (2, 8), dtype=torch.uint8)

    p = pack_2bit(x)
    y = unpack_2bit(p)

    print(x)
    print(p)
    print(torch.equal(x, y))  # True


    x = torch.randint(0, 8, (2, 16), dtype=torch.uint8)

    p = pack_3bit(x)
    y = unpack_3bit(p)

    print(x)
    print(p)
    print(torch.equal(x, y))  # True


    x = torch.randint(0, 16, (2, 8), dtype=torch.uint8)

    p = pack_4bit(x)
    y = unpack_4bit(p)

    print(x)
    print(p)
    print(torch.equal(x, y))  # True
