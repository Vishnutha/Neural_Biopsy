import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

for num in range(1, 82):
    patch1 = Image.open(f"patches_512x512/systematic_patch_{num}.png")
    patch_np = np.array(patch1)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    x_ticks = np.arange(0, 513, 20)
    y_ticks = np.arange(0, 513, 20)

    _, thresholded = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 6)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    eroded_image = cv2.erode(closed, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eroded_image, (7, 7), 0)

    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    output_image_bw = patch_np
    min_contour_area = 320
    object_counter = 1
    radii_data = []
    rbc = 0
    directory_name = "direct"

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_contour_area:
            continue

        if hierarchy[0][i][3] == -1:
            (x_outer, y_outer), outer_radius = cv2.minEnclosingCircle(contour)
            cv2.drawContours(output_image_bw, [contour], -1, (255, 0, 0), 2)
            cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_outer) - 10, int(y_outer) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            inner_radius = 0
            largest_inner_contour = None
            max_area = 0
            inner_radius = 0

            for j, child_contour in enumerate(contours):
                if hierarchy[0][j][3] == i:
                    area = cv2.contourArea(child_contour)
                    if area > max_area:
                        max_area = area
                        largest_inner_contour = child_contour
                        (x_inner, y_inner), inner_radius = cv2.minEnclosingCircle(child_contour)

            if largest_inner_contour is not None:
                cv2.drawContours(output_image_bw, [largest_inner_contour], -1, (0, 255, 0), 2)
                cv2.putText(output_image_bw, f"Obj {object_counter}", (int(x_inner) - 10, int(y_inner) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(output_image_bw, [contour], -1, (0, 255, 0), 2)
                cv2.putText(output_image_bw, "RBC", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                rbc += 1

            thickness = outer_radius - inner_radius
            scale_factor = 0.136
            diameter = 2 * outer_radius
            radii_data.append({
                "axon_id": object_counter,
                "outer_radius": outer_radius * scale_factor,
                "inner_radius": inner_radius * scale_factor,
                "thickness": thickness * scale_factor,
                "diameter": diameter*scale_factor,
                "center": (x_outer, y_outer)
            })
            object_counter += 1

    percentage = (rbc / object_counter) * 100
    if percentage > 55:
        directory_name = "rbc"
    print(percentage)

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[0].imshow(patch1, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title("Blur Thresholded Image")
    axes[1].axis('off')
    axes[2].imshow(output_image_bw)
    axes[2].set_title("Output Image with Contours")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/{directory_name}/output_image_dir_{num}', dpi=300)
    print(f"Subplots saved to results/{directory_name} _ {num}")

    for axon_data in radii_data:
        obj_id = axon_data["axon_id"]
        center = axon_data["center"]
        outer_radius = axon_data["outer_radius"]
        inner_radius = axon_data["inner_radius"]
        thickness = axon_data["thickness"]
        diameter = axon_data["diameter"]
        
        print(f"Object {obj_id}:")
        print(f"  Center = {center[0]:.2f} , {center[1]:.2f}")
        print(f"  Outer Radius = {outer_radius:.4f} micro meters")
        print(f"  Diameter = {diameter:.4f} micro meters")
        print(f"  Inner Radius = {inner_radius:.4f} micro meters")
        print(f"  Thickness = {thickness:.4f} micro meters\n")

