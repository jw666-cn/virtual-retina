import cv2, numpy as np
from pathlib import Path
out = Path("artifacts/dynamic"); out.mkdir(parents=True, exist_ok=True)
w,h,fps,sec = 640,360,25,10
vw = cv2.VideoWriter(str(out/"motion.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
for i in range(fps*sec):
    f = np.zeros((h,w,3), np.uint8); f[:] = (18,24,32)
    t = i/fps
    x = int(w*0.1 + (w*0.8)*abs(((t*0.5)%2)-1))
    cv2.circle(f, (x, int(h*0.45)), 38, (40,40,235), -1)
    cv2.rectangle(f, (int(w*0.62), int(h*0.66)), (int(w*0.62)+120, int(h*0.66)+70), (60,180,90), -1)
    cv2.putText(f, f"VIDEO t={t:05.2f}s", (18, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230,230,230), 2)
    vw.write(f)
vw.release()
print("mp4", (out/"motion.mp4").stat().st_size)
