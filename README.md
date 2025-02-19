# Ultralytics

## Boxes
Use the Following Code to Run the Ultralytics Box Feature
```
from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on 'bus.jpg'
results = model("https://ultralytics.com/images/bus.jpg", verbose=True)

# Visualize the results
for i, r in enumerate(results):
    print(f'{i=}, {r=}')
          
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
```
Use ``` results[0].boxes ``` to focus on the following lists:

cls: Identifies each of the objects identified in the image based on a dictionary that assigns a number to an item. For example, it returned **cls: tensor([5., 0., 0., 0., 0.]),** meaning it identified 5 objects as there are five numbers listed. It is also saying that the first object was a bus and the other 4 were a people, These numbers came from the dictionary called names that was listed in the results.

conf: This tells you the confidence level of identifying the object on a range of 0 to 1, where 1 is the most confidence. For example, the code reported in the results **conf: tensor([0.9402, 0.8882, 0.8783, 0.8558, 0.6219])**. Thus, we can conclude that it was able to identify the bus with the most confidence of 0.9402 but the other 4 people with a confidence of 0.8882, 0.8783, 0.8558 and 0.6219 respectively.

data: This tells you the top left x and y values, the bottom right x and y values, the confidence in detection, and the class. The confidence were found previously in cls and conf lists. The x and y values tell us the exact coordinates in pixels of where the top left and bottom right corners of the box are in respect to the origin of the image(top left corner). For example, based on the results, **data: tensor([[3.8327e+00, 2.2936e+02, 7.9619e+02, 7.2841e+02, 9.4015e-01, 5.0000e+00], [6.7102e+02, 3.9483e+02, 8.0981e+02, 8.7871e+02, 8.8822e-01, 0.0000e+00], [4.7405e+01, 3.9957e+02, 2.3930e+02, 9.0420e+02, 8.7825e-01, 0.0000e+00], [2.2306e+02, 4.0869e+02, 3.4447e+02, 8.6044e+02, 8.5577e-01, 0.0000e+00], [2.1726e-02, 5.5607e+02, 6.8885e+01, 8.7236e+02, 6.2192e-01, 0.0000e+00]])** we can see that for the bus the top left corner of the box is located at 3.83 pixels along the x axis and 229.36 pixels along the y axis.

xywh: Provides the absolute coordinates in pixels of the center of your object. The dimensions are x- value, y-value, width and height. For example, **xywh: tensor([[400.0136, 478.8883, 792.3618, 499.0480], [740.4135, 636.7728, 138.7925, 483.8793], [143.3527, 651.8801, 191.8959, 504.6299], [283.7633, 634.5621, 121.4087, 451.7472], [ 34.4536, 714.2139,  68.8637, 316.2906]]).**  From this we can see that the center of the bus is located at 400.01 pixels on the x-axis and 478.89 pixel on the y-axis. It also has a width of 792.36 pixels and height of 499.05 pixels.

xywhn: Provides the size of the object relative to the image on scale of 0 to 1, where 1 means it takes up the full length of that dimension. For example, using the results from the image **xywhn: tensor([[0.4938, 0.4434, 0.9782, 0.4621], [0.9141, 0.5896, 0.1713, 0.4480], [0.1770, 0.6036, 0.2369, 0.4672], [0.3503, 0.5876, 0.1499, 0.4183], [0.0425, 0.6613, 0.0850, 0.2929]])**, we can see that the center of the bus is at 0.4938 along the x axis and at 0.4434 along the y-axis and it has a width that takes up 97.82% of the image horizontally and a height that takes up 46.21% of image vertically.

xyxy: Provides the absolute coordinates in pixels of the top left x and y values and the bottom right x and y values of your box. For example, from the results it says **xyxy: tensor([[3.8327e+00, 2.2936e+02, 7.9619e+02, 7.2841e+02], [6.7102e+02, 3.9483e+02, 8.0981e+02, 8.7871e+02], [4.7405e+01, 3.9957e+02, 2.3930e+02, 9.0420e+02],[2.2306e+02, 4.0869e+02, 3.4447e+02, 8.6044e+02],[2.1726e-02, 5.5607e+02, 6.8885e+01, 8.7236e+02]])** we can see that for the bus the top left corner of the box is located at 3.83 pixels along the x axis and 229.36 pixels along the y axis.

xyxyn: Provides the of the box relative to the image on scale of 0 to 1, where 1 means it takes up the full length of that dimension. It tells us where the top left and bottom right corners of the box are in respect to the origin of the image(top left corner).For example, using the results from the image **xyxyn: tensor([[4.7318e-03, 2.1237e-01, 9.8296e-01, 6.7446e-01], [8.2842e-01, 3.6559e-01, 9.9977e-01, 8.1362e-01],[5.8524e-02, 3.6997e-01, 2.9543e-01, 8.3722e-01], [2.7538e-01, 3.7842e-01, 4.2527e-01, 7.9670e-01], [2.6822e-05, 5.1488e-01, 8.5044e-02, 8.0774e-01]])**, we can conclude that the top left corner of the box is at 0.47% in width and 21.24% in height. 

## Pose/Keypoints
Use the Following Code to Run the Ultralytics Keypoints Feature

```
from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n-pose.pt")

# Run inference on 'bus.jpg'
results = model("https://ultralytics.com/images/bus.jpg", verbose=True)

# Visualize the results
for i, r in enumerate(results):
    print(f'{i=}, {r=}')
          
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
```
Use ``` results[0].keypoints ``` to focus on the following lists:

conf: This provides a confidence level on a scale of 0 to 1, 1 being the most confident, on how confident it was able to identify a keypoint. Each of the values shown correspond to a keypoint in the body identified shown in the index below:

**0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear
5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist
11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle**

Thus from reading the results, 
**conf: tensor([[0.9910, 0.9289, 0.9869, 0.4267, 0.9313, 0.9907, 0.9976, 0.9248, 0.9884, 0.9015, 0.9744, 0.9969, 0.9984, 0.9949, 0.9975, 0.9785, 0.9856],
        [0.1586, 0.1561, 0.0468, 0.2351, 0.0505, 0.6711, 0.2402, 0.5964, 0.1104, 0.4541, 0.1319, 0.7288, 0.5135, 0.7590, 0.5564, 0.5935, 0.4370],
        [0.9894, 0.9335, 0.9794, 0.5549, 0.9086, 0.9952, 0.9976, 0.9465, 0.9778, 0.9134, 0.9481, 0.9983, 0.9987, 0.9954, 0.9964, 0.9774, 0.9804],
        [0.0987, 0.0392, 0.0631, 0.0392, 0.0677, 0.2103, 0.2339, 0.2615, 0.3053, 0.3423, 0.3554, 0.2780, 0.2918, 0.2393, 0.2481, 0.1393, 0.1388]])** 
It can be concluded that it was able to identify the nose in the first person with 99.1% confidence

data: This tells you the x-value and y-value in pixels of each keypoint along with the confidence level of the identification of the keypoint on a scale from 0 to 1. Based on the following data, it can be concluded that the nose key point is located at 142.36 pixels along the x-axis and 441.86 pixels along the y-axis with a confidence of 99.1% for the first person.

## Segment/Mask

Use the Following Code to Run the Ultralytics Segment Feature

```
from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n-seg.pt")

# Run inference on 'bus.jpg'
results = model("https://ultralytics.com/images/bus.jpg", verbose=True)

# Visualize the results
for i, r in enumerate(results):
    print(f'{i=}, {r=}')
          
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
```
Use ``` results[0].masks.data[0][240,:] ``` to focus on the following lists:

data: This data tells us whether a specific pixel on an image was masked(1) or was a part of the background(0) using binary code. In the code shown above, we are only going to observe row 240 of the image. As a result of that code, we recieved the following results

**tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])**

## OBB

Use the Following Code to Run the Ultralytics OBB Feature


```
from PIL import Image

from ultralytics import YOLO

# Obb:
model = YOLO("yolo11n-obb.pt")
results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

# Visualize the results
for i, r in enumerate(results):
    print(f'{i=}, {r=}')
          
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
```
Use ``` results[0].obb ``` to focus on the following lists:

cls: Identifies each of the objects identified in the image based on a dictionary that assigns a number to an item. For example, when we recieved the following results from the code:

**cls: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 7., 1., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 7., 1., 1., 1., 1., 1., 1.])**

we were able to determine from the indicies that 1 means it is ship that it identifies and 7 means that a harbor has been identified.

conf: This tells you the confidence level of identifying the object on a range of 0 to 1, where 1 is the most confidence. For example, the code reported in the results:
**conf: tensor([0.8425, 0.8336, 0.8319, 0.8297, 0.8263, 0.8244, 0.8243, 0.8218, 0.8197, 0.8181, 0.8177, 0.8162, 0.8147, 0.8134, 0.8132, 0.8122, 0.8106, 0.8105, 0.8101, 0.8097, 0.8094, 0.8076, 0.8074, 0.8067, 0.8067, 0.8067, 0.8065, 0.8061, 0.8059, 0.8055, 0.8052, 0.8035, 0.8026, 0.8014, 0.7996, 0.7991, 0.7979, 0.7965, 0.7965,
        0.7909, 0.7904, 0.7895, 0.7891, 0.7887, 0.7887, 0.7884, 0.7861, 0.7851, 0.7846, 0.7845, 0.7843, 0.7841, 0.7840, 0.7838, 0.7835, 0.7817, 0.7807, 0.7806, 0.7804, 0.7777, 0.7772, 0.7754, 0.7744, 0.7742, 0.7736, 0.7720, 0.7711, 0.7696, 0.7692, 0.7669, 0.7664, 0.7652, 0.7638, 0.7634, 0.7625, 0.7595, 0.7591, 0.7493,
        0.7490, 0.7480, 0.7475, 0.7473, 0.7471, 0.7468, 0.7462, 0.7456, 0.7430, 0.7369, 0.7368, 0.7358, 0.7353, 0.7345, 0.7341, 0.7340, 0.7278, 0.7273, 0.7270, 0.7266, 0.7253, 0.7252, 0.7224, 0.7213, 0.7201, 0.7196, 0.7160, 0.7131, 0.7102, 0.7079, 0.7063, 0.7025, 0.7014, 0.7007, 0.6997, 0.6957, 0.6889, 0.6873, 0.6849,
        0.6839, 0.6818, 0.6815, 0.6803, 0.6785, 0.6743, 0.6734, 0.6732, 0.6727, 0.6726, 0.6705, 0.6690, 0.6682, 0.6673, 0.6632, 0.6568, 0.6557, 0.6507, 0.6438, 0.6435, 0.6367, 0.6336, 0.6320, 0.6290, 0.6271, 0.6255, 0.6169, 0.6096, 0.5993, 0.5991, 0.5949, 0.5922, 0.5855, 0.5845, 0.5666, 0.5582, 0.5355, 0.5275, 0.5091,
        0.4968, 0.4921, 0.4763, 0.4447, 0.4364, 0.4333, 0.4177, 0.4068, 0.4049, 0.3979, 0.3959, 0.3898, 0.3788, 0.3784, 0.3600, 0.3593, 0.3432, 0.3253, 0.2982, 0.2772, 0.2572])**

is telling us that the first identified ship has a confidence level of 84.25%

xyxyxyxy:  it gives the coordinates(x,y) of all four corners of each box in pixels

## Tracking ID

Use the Following Code to Run the Ultralytics Track ID Feature


```
from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Tracking:
results = model.track("https://ultralytics.com/images/bus.jpg", verbose=True)

# Visualize the results
for i, r in enumerate(results):
    print(f'{i=}, {r=}')
          
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
```
Use ``` results``` to focus on the following lists:

In the returned image it will give each identified object an ID number
  
