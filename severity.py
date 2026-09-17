from PIL import Image
import numpy as np

class SeverityEstimator:
    """Baseline estimator. Production severity should be replaced by a disease-mask model trained on field data."""
    def estimate(self, image: Image.Image) -> dict:
        arr=np.asarray(image.convert('RGB').resize((512,512))).astype('float32')/255
        r,g,b=arr[...,0],arr[...,1],arr[...,2]
        # conservative visual proxy for brown/yellow lesion pixels; not a clinical/agronomic measurement
        lesion=((r>0.28)&(r>g*1.08)&(g<0.72)&(b<0.60)) | ((r>0.45)&(g>0.35)&(g<0.75)&(b<0.30))
        green=(g>r*0.9)&(g>b*1.05)
        denom=max(int(green.sum()),1)
        pct=float(np.clip(100*lesion.sum()/denom,0,100))
        return {'severity_percent_proxy':round(pct,2),'method':'color-proxy','validated':False}
