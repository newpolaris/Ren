> Memory Type

VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT - near GPU, fastest
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT - access from CPU write even read
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT - whenever we do write from a CPU memory and then submit
                                       a comment to the gpu in a command buffer,
                                       you don't need to do extra operations to make sure this memory is like visible
VK_MEMORY_PROPERTY_HOST_CACHED_BIT - gpu fills buffer and you want to read that 

> 'Optimize mesh rendering'

0:03:00 - triangle 1087474 in 2.3ms result is not fast, because environment is RTX 2080
0:08:57 - test present mode immediate, disable V-sync
0:29:21 - in a real vulkan application you need some sort of upload syn management might to use transfer queues
         (http://lifeisforu.tistory.com/404?category=837815 에서는 공용 17개, 전송 전용이 1개. 전용이 존재하는 경우 그것을 사용한다고 했다)
0:40:45 - Device Local Memory make 0.31 ms
0:43:42 - Draw 10 times, makes mesh shader 10.41 vs 2.3 ms (stroage buffer or FVF)
0:54:26 - Test optimizeVertexCache with randome suffle
1:02:32 - Test push descriptor with position * 1e-9. 1.9 ms > 1.0 ms. 
          레스터한것과 안한것의 결과가 2배정도면 거의 optimal 로 불수 있지 않을까? (position을 0 및 0.1로 둬서 생성을 막았다)
          5 ~ 10 billion 이 결과 예상값이고 5 정도 나오면 그 결과에 만족할 수 있음.
1:04:20 - N triangle > n/2 vertices (minecraft 같은건 nx2)
          upto 128 and one is used by gl_primitiveCountNV (so, 127) and makes divided by 3. up to 42 triangles
1:09:12 - vertex data: 16 KB / meshlet & 64 vertices -> 16 * 1024 / 64 -> 256 byte per vertex
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

> 'Meshlet culling'

1:19:01 - cone-culling-experments-log 의 결과를 base로 삼아서 비교, 거기서 budda는 24% 정도가 나옴
          기본 설정에서는 meshlet은 3% 정도가 나옴. triangleCount를 줄여서 41으로 하니 10% 정도로 올라감



- happy_buddha, Nvidia 660m, one commandbuffer with wait-idle

- FVF vs PUSH descriptor

            'basic' 'duplicate remove' 'optimize vertex fetch' 'reduce vertex size'
    FVF  :   2.54          1.0                0.84                     0.84
    PUSH :   2.67          3.8                1.25                     0.91


- indirect drawing testing with CPU Culling on cluster meshlet

    run x5 

    Single  / Multiple / Indirect
    4.37 ms / 5.75 ms  / 5.39 ms (1)
    5.77 ms / 5.55 ms  / 5.40 ms (2)

    (1) 메쉬렛 생성의 고정 상수 및 몇개 구문 바꾸기 전 (멀 잘못 고친 버그 요인이라 생각)
    (2) 뭘 바ㅤㄲㅝㅅ는데 그다음부터 이와 같은 결과가 나옴

    문제는 vertex/fragment의 작업이 지나치게 단순하다는 것과 
    multiple indirect 의 성능이 더 낮거나 유사하다는 것
    그리고 scene 자체가 너무 단순히 가운데 객체가 있다는 것
    64개의 버텍스가 옵티멀이 아닐 수 있다는 점. (1) 테스트 할 때는 128 부터가 가장 best 였으나 (2) 에서는 64가 최적;

    (https://gist.github.com/zeux/1cef1417215add13c9eb26451f26afe2) 여기선 128 부터 떨어진다고 되어있음

- 5x5 buddha, Nvida 660m, 10만 삼각형, one commandbuffer with wait command queue
 
          'basic' 'optimize lib' '16bit vertex fetch' 
    FVF :  0.2b        0.7b              0.7b 
    PUSH:  0.2b        0.4b              0.7b

    10x10 으로 바꿨을 때 0.8 / 추가로 texture coordinate 없앴지만 0.8 로 동일하다.
    cluster culling의 경우 kitty는 20% 가 줄어서 1.0b 까지 올랐으나,
                           budda는 meshopt 로 수행했음에도 3% 미만만 준다. 0.8b 까지 됨

                           

