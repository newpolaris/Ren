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
         (http://lifeisforu.tistory.com/404?category=837815 ������ ���� 17��, ���� ������ 1��. ������ �����ϴ� ��� �װ��� ����Ѵٰ� �ߴ�)
0:40:45 - Device Local Memory make 0.31 ms
0:43:42 - Draw 10 times, makes mesh shader 10.41 vs 2.3 ms (stroage buffer or FVF)
0:54:26 - Test optimizeVertexCache with randome suffle
1:02:32 - Test push descriptor with position * 1e-9. 1.9 ms > 1.0 ms. 
          �������ѰͰ� ���Ѱ��� ����� 2�������� ���� optimal �� �Ҽ� ���� ������? (position�� 0 �� 0.1�� �ּ� ������ ���Ҵ�)
          5 ~ 10 billion �� ��� �����̰� 5 ���� ������ �� ����� ������ �� ����.
1:04:20 - N triangle > n/2 vertices (minecraft ������ nx2)
          upto 128 and one is used by gl_primitiveCountNV (so, 127) and makes divided by 3. up to 42 triangles
1:09:12 - vertex data: 16 KB / meshlet & 64 vertices -> 16 * 1024 / 64 -> 256 byte per vertex
          index data: 128 byte chuck (4 bytes for count of primitive) and 1 byte per index (so limited to 2^8 = 256))
          if use 128x3 then (128x3 - 4) index > 126 triangles
          if use 128x2 then (128x2 - 4) index >  84 triangles
          if use 128x1 then (128x1 - 4) index >  41 triangles (ndivia article�� 40�̶� ����; introduction to utrning mehs shader)

1:39:12 - perprimitiveNV �� primitive �� vec3 ���� �ѱ� �� �ִ�.
2:10:15 - ACMR ratio (average cache miss ratio)  = # of vertex shader invocation / # of triangles         
          0.5 best case
          0.7 game mesh ~ 1.5 light map terribly packed
          2.0 minecraft like
2:14:48 - Nvidia fixed function rasterization pipleline at least 1080 not 2080 necessily, 
          as having 32 vertices meshlet. Gpu accumulate wraps upto 32 vertices. cont.
          32 = acmr 0.75
          64 = acmr 0.65

> 'Meshlet culling'

1:19:01 - cone-culling-experments-log �� ����� base�� ��Ƽ� ��, �ű⼭ budda�� 24% ������ ����
          �⺻ ���������� meshlet�� 3% ������ ����. triangleCount�� �ٿ��� 41���� �ϴ� 10% ������ �ö�



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

    (1) �޽��� ������ ���� ��� �� � ���� �ٲٱ� �� (�� �߸� ��ģ ���� �����̶� ����)
    (2) �� �٤Ԥ��ͤ��µ� �״������� �̿� ���� ����� ����

    ������ vertex/fragment�� �۾��� ����ġ�� �ܼ��ϴٴ� �Ͱ� 
    multiple indirect �� ������ �� ���ų� �����ϴٴ� ��
    �׸��� scene ��ü�� �ʹ� �ܼ��� ��� ��ü�� �ִٴ� ��
    64���� ���ؽ��� ��Ƽ���� �ƴ� �� �ִٴ� ��. (1) �׽�Ʈ �� ���� 128 ���Ͱ� ���� best ������ (2) ������ 64�� ����;

    (https://gist.github.com/zeux/1cef1417215add13c9eb26451f26afe2) ���⼱ 128 ���� �������ٰ� �Ǿ�����

- 5x5 buddha, Nvida 660m, 10�� �ﰢ��, one commandbuffer with wait command queue
 
          'basic' 'optimize lib' '16bit vertex fetch' 
    FVF :  0.2b        0.7b              0.7b 
    PUSH:  0.2b        0.4b              0.7b

    10x10 ���� �ٲ��� �� 0.8 / �߰��� texture coordinate �������� 0.8 �� �����ϴ�.
    cluster culling�� ��� kitty�� 20% �� �پ 1.0b ���� �ö�����,
                           budda�� meshopt �� ������������ 3% �̸��� �ش�. 0.8b ���� ��

                           

