> Optimize mesh rendering

003:00 - triangle 1087474 in 2.3ms result is not fast, because envrionment is RTX 2080
008:57 - test present mode immediate, disable V-sync
029:21 - in a real vulkan application you need some sort of upload syn management might to use transfer queues
         (http://lifeisforu.tistory.com/404?category=837815 에서는 공용 17개, 전송 전용이 1개. 전용이 존재하는 경우 그것을 사용한다고 했다)
040:45 - Device Local Memory make 0.31 ms
043:42 - Draw 10 times, makes mesh shader 10.41 vs 2.3 ms (stroage buffer or FVF)
054:26 - Test optimizeVertexCache with randome suffle

happy_buddha, Nvidia 660m, one commandbuffer with wait-idle

        'basic' 'duplicate remove' 'optimize vertex fetch'
FVF  :   2.54          1.0                0.84
PUSH :   2.67          3.8                1.25




Worklist:

* FVF (fixed function vertex fuction)
* Device Local Memory


