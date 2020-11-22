from scripts.predict import run
from skimage import io

df , ans , img = run('~/Downloads/WhatsApp Image 2020-11-22 at 19.45.12.jpeg')

print(df)
print('Answer: ',ans)
#io.imsave('~/Downloads/zx.png', img)