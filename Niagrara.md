> Memory Type

VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT - near GPU, fastest
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT - access from CPU write even read
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT - whenever we do write from a CPU memory and then submit
                                       a comment to the gpu in a command buffer,
                                       you don't need to do extra operations to make sure this memory is like visible
VK_MEMORY_PROPERTY_HOST_CACHED_BIT - gpu fills buffer and you want to read that 

> 'Optimize mesh rendering'

0:03:00 - triangle 1087474 in 2.3ms result is not fast, because envrionment is RTX 2080
0:08:57 - test present mode immediate, disable V-sync
0:29:21 - in a real vulkan application you need some sort of upload syn management might to use transfer queues
         (http://lifeisforu.tistory.com/404?category=837815 에서는 공용 17개, 전송 전용이 1개. 전용이 존재하는 경우 그것을 사용한다고 했다)
0:40:45 - Device Local Memory make 0.31 ms
0:43:42 - Draw 10 times, makes mesh shader 10.41 vs 2.3 ms (stroage buffer or FVF)
0:54:26 - Test optimizeVertexCache with randome suffle
1:02:32 - Test push descriptor with position * 1e-9. 1.9 ms > 1.0 ms. 레스터한것과 안한것의 결과가 2배정도면 거의 optimzal 로 본다. 5 ~ 10 billion 이 결과 예상값이고 5 정도 나오면 그 결과에 만족할 수 있음.
1:04:20 - N triangle > n/2 vertices (minecraft 같은건 nx2)
          upto 128 and one is used by gl_primitiveCountNV (so, 127) and makes divided by 3. up to 42 triangles
1:09:12 - vertex data: 16 KB / meshlet & 64 vertices > 16 * 1024 / 64 > 256 byte per vertex
          index data: 128 byte chuck (4 bytes for count of primitive) and 1 byte per index (so limited to 2^8 = 256))
          if use 128x3 then (128x3 - 4) index > 126 triangles
          if use 128x2 then (128x2 - 4) index >  84 triangles
          if use 128x1 then (128x1 - 4) index >  41 triangles (ndivia article은 40이라 말함; introduction to utrning mehs shader)

1:39:12 - perprimitiveNV 로 primitive 별 vec3 등을 넘길 수 있다.
2:10:15 - ACMR ratio (average cache miss ratio)  = # of vertex shader invocation / # of triangles         
          0.5 best case
          0.7 game mesh ~ 1.5 light map terribly packed
          2.0 minecraft like
2:14:48 - Nvidia fixed function rasterization pipleline at least 1080 not 2080 necessily, 
          as having 32 vertices meshlet. Gpu accumulate wraps upto 32 vertices. cont.
          32 = acmr 0.75
          64 = acmr 0.65



happy_buddha, Nvidia 660m, one commandbuffer with wait-idle

- FVF vs PUSH descriptor

        'basic' 'duplicate remove' 'optimize vertex fetch' 'reduce vertex size'
FVF  :   2.54          1.0                0.84                     0.84
PUSH :   2.67          3.8                1.25                     0.91


- indirect drawing testing with CPU Culling

run x5 

PUSH    / Draw    / Indirect
4.37 ms / 5.75 ms / 5.39 ms




Worklist:

* Device Local Memory
