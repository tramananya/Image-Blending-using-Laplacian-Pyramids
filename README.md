# Image-Blending-using-Laplacian-Pyramids
Implemention of Laplacian and Gaussian pyramids using 2D convolution and a GUI to select ROI for image blending using Python and OpenCV.


Following is the brief description of the functions I have written in the code:  
- Gaussianfilter: Creates the Gaussian filter for given sigma 
- Conv2: This is the function for convolution 
- Upscale: upscales the given image to new size 
- Downscale: downscales the given image to required level 
- Build_gaussian: creates the Gaussian pyramid
- Build_laplacian : creates the Laplacian pyramid 
- Draw_circle: used to read the mouse pointer reading and draw small circles on the points for visualization.   
- Selectroi:  This is the function that selects the ROI.                         
    NOTE1: To select the ROI please click once and drag as much as you want. Whenever you want to end the selection, click again. You don’t have to keep the mouse clicked. 
- Blend_images: This is where the blending happens, where the main equation for blending is implemented.  
- Create_image: This function takes care of padding zeros to smaller foreground images and returns padded image. It also aligns the image in the right place. 

Following is the description for the way the code works:  
1. The code asks, Do you want to run examples? If you want to see the 4 examples I have implemented, please type 1.  
2. The next question, if you chose 1 for previous question is which example. Choose any number from 1 to 4.  
3. Next, it will ask for image type: color or gray 
4. Next, pad type: for convolution 
5. Next, number of layers 
6. A screen pops up, select the ROI on it as described in the NOTE1.  
7. In some time, the blended image pops up.  


NOTE: All the images have been taken from google and upsclaed/downsclaed as required.
