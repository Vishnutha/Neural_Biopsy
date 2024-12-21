


# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the ima
# image = cv2.imread('data/set1_Image_01_40x_bf_02_fossicle_binary.png')
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# from PIL import Image
# import matplotlib.pyplot as plt

# # Load the image
# image_path = "data/set1_Image_01_40x_bf_02_fossicle.png"  # Update the path to your image
# image = Image.open(image_path)

# # Define patch size
# patch_width = 512  # Width of each patch
# patch_height = 512  # Height of each patch

# # Get the image dimensions
# image_width, image_height = image.size

# # Calculate the number of patches that fit horizontally and vertically
# num_patches_x = image_width // patch_width
# num_patches_y = image_height // patch_height

# # Create a list to store patches
# patches = []

# # Generate patches systematically
# for row in range(num_patches_y):
#     for col in range(num_patches_x):
#         # Define the top-left corner of the patch
#         x_start = col * patch_width
#         y_start = row * patch_height
        
#         # Define the bottom-right corner of the patch
#         x_end = x_start + patch_width
#         y_end = y_start + patch_height
        
#         # Crop the patch from the image
#         patch = image.crop((x_start, y_start, x_end, y_end))
        
#         # Append the patch and its coordinates to the list
#         patches.append({
#             "coordinates": (x_start, y_start, x_end, y_end),
#             "image": patch
#         })

# # Display the first 5 patches for verification
# for idx, patch_info in enumerate(patches[:5]):
#     patch_image = patch_info["image"]
#     coords = patch_info["coordinates"]
    
#     plt.figure(figsize=(5, 5))  # Adjust the figure size
#     plt.imshow(patch_image, cmap='gray')
#     plt.title(f"Patch {idx + 1} - Coordinates: {coords}")
#     plt.axis("on")
#     plt.show()
# Save each patch as a PNG file


# %% [markdown]
# # Patch saving 

# %%

# # Optional: Save patches as separate files
# for idx, patch_info in enumerate(patches):
#     patch_image = patch_info["image"]
#     patch_image.save(f"systematic_patch_{idx + 1}_{x_start}_{y_start}.png")  

# %% [markdown]
# # Manual Patch Generation

# %%
# # Import necessary libraries
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# # Load the image from the notebook
# image_path = "data/set1_Image_01_40x_bf_02_fossicle.png"  # Update this if your image is stored separately

# # Open the image (update the file reading as per your stored image format)
# image = Image.open(image_path)

# # Define the coordinates for cropping
# x_start1, x_end1 = 1450, 1750
# y_start1, y_end1 = 2250, 2600

# # Crop the image to the desired patch
# patch1 = image.crop((x_start1, y_start1, x_end1, y_end1))

# %%
# Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
for num in range(1,82):
# patch_path =  
    patch1 = Image.open(f"patches_512x512/systematic_patch_{num}.png")
    patch_np = np.array(patch1)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    x_ticks = np.arange(0,513,20)  # Tick positions for the x-axis
    y_ticks = np.arange(0,513, 20)  # Tick positions for the y-axis

    # Display the patch with axes
    # plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    # plt.imshow(patch1, cmap='gray')
    # plt.title("Cropped Patch with Axes and Tick Intervals of 20 Pixels")
    # plt.xlabel("X-axis (pixels)")
    # plt.ylabel("Y-axis (pixels)")
    # plt.axis("on")  # Turn on axes
    # plt.xticks(ticks=np.linspace(0, patch1.size[0], len(x_ticks)), labels=x_ticks)  # X-axis ticks
    # plt.yticks(ticks=np.linspace(0, patch1.size[1], len(y_ticks)), labels=y_ticks)  # Y-axis ticks
    # plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)  # Optional grid for better visualization
    # plt.show()

    # %%
    _, thresholded = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,27,6)
 


    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Apply morphological closing to fix gaps in the binary image
    kernel = np.ones((3,3), np.uint8)  # Adjust kernel size if needed
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(closed, kernel, iterations=1) 
    blurred = cv2.GaussianBlur(eroded_image, (7, 7), 0)

    # plt.show()
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    # Create a copy of the binary image for visualization
    output_image_bw = patch_np

    # Define a minimum area to filter out noise
    min_contour_area = 320 # Adjust as needed based on your image

    # Object counter
    object_counter = 1

    # Data to store radii information
    radii_data = []
    rbc = 0
    directory_name = "direct"
    # Loop through contours to classify as inner or outer boundaries
    for i, contour in enumerate(contours):
        # Filter out small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Check if the contour is an outer boundary (no parent in the hierarchy)
        if hierarchy[0][i][3] == -1:  # Outer boundary
            # Calculate the outer radius
            (x_outer, y_outer), outer_radius = cv2.minEnclosingCircle(contour)

            # Draw the outer boundary in blue
            cv2.drawContours(output_image_bw, [contour], -1, (255, 0, 0), 2)  # Blue for outer
            # Add object number as label
            cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_outer) - 10, int(y_outer) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Look for corresponding inner boundary (child in the hierarchy)
            inner_radius = 0  # Default if no inner boundary is found
            # Initialize variables to store the largest child contour and its radius
            largest_inner_contour = None
            max_area = 0
            inner_radius = 0
            
            # Loop through all child contours to find the one with the largest area
            for j, child_contour in enumerate(contours):
                if hierarchy[0][j][3] == i:  # Check if the current contour is a child of the current outer contour
                    # Calculate the area of the child contour
                    area = cv2.contourArea(child_contour)
                    if area > max_area:  # Update the largest contour if this one is larger
                        max_area = area
                        largest_inner_contour = child_contour
                        (x_inner, y_inner), inner_radius = cv2.minEnclosingCircle(child_contour)
            
            # After the loop, use the largest child contour (if found)
            if largest_inner_contour is not None:
                # Draw the inner boundary in green
                cv2.drawContours(output_image_bw, [largest_inner_contour], -1, (0, 255, 0), 2)  # Green for inner
                # Add the same object number to the inner boundary
                cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_inner) - 10, int(y_inner) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else :
                x, y, w, h = cv2.boundingRect(contour)
            # Draw the contour
                cv2.drawContours(output_image_bw, [contour], -1, (0, 255, 0), 2)  # Green contour
            # Annotate with "RBC"
                cv2.putText(output_image_bw, "RBC", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                rbc += 1
            # Calculate thickness if both rdii exist
            thickness = outer_radius - inner_radius
            scale_factor = 0.136
            # radii_data.append((object_counter, outer_radius*scale_factor, inner_radius*scale_factor, thickness*scale_factor,(x_outer,y_outer))
            diameter = 2*outer_radius
            # radii_data.append((object_counter, outer_radius, inner_radius, thickness))
            radii_data.append({
                "axon_id": object_counter,  # Unique ID for the axon
                "outer_radius": outer_radius * scale_factor,  # Outer radius scaled
                "inner_radius": inner_radius * scale_factor,  # Inner radius scaled
                "thickness": thickness * scale_factor,
                "diameter" : diameter,
                "center": (x_outer,y_outer)# Thickness scaled
            })
            # Increment the object counter for the next object
            object_counter += 1
    percentage = (rbc/object_counter)*100
    if(percentage > 55):
        directory_name = "rbc"
    print(percentage)
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    axes[ 0].imshow(patch1, cmap='gray')
    axes[ 0].set_title("Input Image")
    axes[ 0].axis('off')

    axes[1].imshow(blurred, cmap='gray')
    axes[ 1].set_title("Blur Thresholded Image")
    axes[ 1].axis('off')

    axes[2].imshow(output_image_bw)
    axes[ 2].set_title("Output Image with Contours")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/{directory_name}/output_image_dir_{num}', dpi=300)
    print(f"Subplots saved to results/{directory_name} _ {num}")
    # plt.imshow(cv2.cvtColor(output_image_bw, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Objects with Inner and Outer Boundaries")
    # plt.axis('off')
    # plt.show()

    # Display the results with numbered objects
    # plt.imshow(cv2.cvtColor(output_image_bw, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Objects with Inner and Outer Boundaries")
    # plt.axis('off')
    # plt.show()

    # Print the outer radius, inner radius, and thickness for each object
    # for obj_id, outer_radius, inner_radius, thickness, _ in radii_data:
    #     print(f"Object {obj_id}:")
    #     print(f"  Outer Radius = {outer_radius:.4f} micro meters")
    #     print(f"  Inner Radius = {inner_radius:.4f} micro meters")
    #     print(f"  Thickness = {thickness:.4f} micro meters\n")
    for axon_data in radii_data:
        obj_id = axon_data["axon_id"]
        center = axon_data["center"]
        outer_radius = axon_data["outer_radius"]
        inner_radius = axon_data["inner_radius"]
        thickness = axon_data["thickness"]
        
        print(f"Object {obj_id}:")
        print(f"  Center = {center[0]:.2f} , {center[1]:.2f}")
        print(f"  Outer Radius = {outer_radius:.4f} micro meters")
        print(f"  Inner Radius = {inner_radius:.4f} micro meters")
        print(f"  Thickness = {thickness:.4f} micro meters\n")



    # %% [markdown]
    # # Sobel 

    # # %%
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Apply morphological closing to fix gaps in the binary image
    # kernel = np.ones((3,3), np.uint8)  # Adjust kernel size if needed
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # eroded_image = cv2.erode(closed, kernel, iterations=1) 
    # blurred = cv2.GaussianBlur(eroded_image, (7, 7), 0)
    # # _, binary_image = cv2.threshold(eroded_image, 127, 255, cv2.THRESH_BINARY)
    # # edges = cv2.Canny(binary_image, 127, 255)
    # sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # X direction
    # sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Y direction

    # # Compute the gradient magnitude
    # sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # # Convert to 8-bit for visualization
    # sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # # plt.imshow(sobel_combined)
    # # plt.title("Sobel Edge Detection")
    # # plt.axis("off")
    # # plt.show()
    # contours, hierarchy = cv2.findContours(sobel_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))
    # # Create a copy of the binary image for visualization
    # output_image_bw = patch_np

    # # Define a minimum area to filter out noise
    # min_contour_area = 320 # Adjust as needed based on your image

    # # Object counter
    # object_counter = 1

    # # Data to store radii information
    # radii_data = []

    # # Loop through contours to classify as inner or outer boundaries
    # for i, contour in enumerate(contours):
    #     # Filter out small contours
    #     if cv2.contourArea(contour) < min_contour_area:
    #         continue

    #     # Check if the contour is an outer boundary (no parent in the hierarchy)
    #     if hierarchy[0][i][3] == -1:  # Outer boundary
    #         # Calculate the outer radius
    #         (x_outer, y_outer), outer_radius = cv2.minEnclosingCircle(contour)

    #         # Draw the outer boundary in blue
    #         cv2.drawContours(output_image_bw, [contour], -1, (255, 0, 0), 2)  # Blue for outer
    #         # Add object number as label
    #         cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_outer) - 10, int(y_outer) - 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #         # Look for corresponding inner boundary (child in the hierarchy)
    #         inner_radius = 0  # Default if no inner boundary is found
    #         # Initialize variables to store the largest child contour and its radius
    #         largest_inner_contour = None
    #         max_area = 0
    #         inner_radius = 0
            
    #         # Loop through all child contours to find the one with the largest area
    #         for j, child_contour in enumerate(contours):
    #             if hierarchy[0][j][3] == i:  # Check if the current contour is a child of the current outer contour
    #                 # Calculate the area of the child contour
    #                 area = cv2.contourArea(child_contour)
    #                 if area > max_area:  # Update the largest contour if this one is larger
    #                     max_area = area
    #                     largest_inner_contour = child_contour
    #                     (x_inner, y_inner), inner_radius = cv2.minEnclosingCircle(child_contour)
            
    #         # After the loop, use the largest child contour (if found)
    #         if largest_inner_contour is not None:
    #             # Draw the inner boundary in green
    #             cv2.drawContours(output_image_bw, [largest_inner_contour], -1, (0, 255, 0), 2)  # Green for inner
    #             # Add the same object number to the inner boundary
    #             cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_inner) - 10, int(y_inner) - 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    #         # Calculate thickness if both rdii exist
    #         thickness = outer_radius - inner_radius
    #         scale_factor = 0.136
    #         # radii_data.append((object_counter, outer_radius*scale_factor, inner_radius*scale_factor, thickness*scale_factor,(x_outer,y_outer)))
            

    # # Sample code to append data (modify the actual implementation accordingly)
    #         # radii_data.append({
    #         #     "axon_id": object_counter,  # Unique ID for the axon
    #         #     "outer_radius": outer_radius * scale_factor,  # Outer radius scaled
    #         #     "inner_radius": inner_radius * scale_factor,  # Inner radius scaled
    #         #     "thickness": thickness * scale_factor,
    #         #     "center": (x_outer,y_outer)# Thickness scaled
    #         # })
    #         diameter = 2*outer_radius
    #         # radii_data.append((object_counter, outer_radius, inner_radius, thickness))
    #         radii_data.append({
    #             "axon_id": object_counter,  # Unique ID for the axon
    #             "outer_radius": outer_radius * scale_factor,  # Outer radius scaled
    #             "inner_radius": inner_radius * scale_factor,  # Inner radius scaled
    #             "thickness": thickness * scale_factor,
    #             "diameter" : diameter,
    #             "center": (x_outer,y_outer)# Thickness scaled
    #         })
    #         # Increment the object counter for the next object
    #         object_counter += 1

    # fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    # axes[ 0].imshow(patch1, cmap='gray')
    # axes[ 0].set_title("Input Image")
    # axes[ 0].axis('off')

    # axes[1].imshow(sobel_combined, cmap='gray')
    # axes[ 1].set_title("Sobel Edge detection Image")
    # axes[ 1].axis('off')

    # axes[2].imshow(output_image_bw)
    # axes[ 2].set_title("Output Image with Contours")
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.savefig(f'results/sobel/output_image_sobel_{num}', dpi=300)
    # print(f"Subplots saved to results/sobel {num} ")
    # # plt.imshow(cv2.cvtColor(output_image_bw, cv2.COLOR_BGR2RGB))
    # # plt.title("Detected Objects with Inner and Outer Boundaries")
    # # plt.axis('off')
    # # plt.show()
    # for axon_data in radii_data:
    #     obj_id = axon_data["axon_id"]
    #     center = axon_data["center"]
    #     outer_radius = axon_data["outer_radius"]
    #     inner_radius = axon_data["inner_radius"]
    #     thickness = axon_data["thickness"]
        
    #     # print(f"Object {obj_id}:")
    #     # print(f"  Center = {center[0]:.2f} , {center[1]:.2f}")
    #     # print(f"  Outer Radius = {outer_radius:.4f} micro meters")
    #     # print(f"  Inner Radius = {inner_radius:.4f} micro meters")
    #     # print(f"  Thickness = {thickness:.4f} micro meters\n")


