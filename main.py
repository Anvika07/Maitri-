<<<<<<< HEAD
import cv2
from PIL import Image
from model_utils import load_model
from inference_utils import predict_emotion


model, feature_extractor = load_model()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    emotion, mood = predict_emotion(model, feature_extractor, img)

    # Display results on frame
    cv2.putText(frame, f"{mood}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Stress/Mood Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

=======
import cv2
from PIL import Image
from model_utils import load_model
from inference_utils import predict_emotion


model, feature_extractor = load_model()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    emotion, mood = predict_emotion(model, feature_extractor, img)

    # Display results on frame
    cv2.putText(frame, f"{mood}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Stress/Mood Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

>>>>>>> 29278773c464dbaaaad8c02d76dc7c4652f8f6fc
