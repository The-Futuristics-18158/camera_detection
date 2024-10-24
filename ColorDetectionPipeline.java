import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Core;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.ArrayList;
import java.util.List;

public class ColorDetectionPipeline extends OpenCvPipeline {

    // Set the YCrCb color ranges for blue based on the slider data
    private final Scalar lowerBlue = new Scalar(55.3, 90.7, 141.7);  // Adjusted blue lower bound
    private final Scalar upperBlue = new Scalar(181.3, 182.8, 255.0); // Adjusted blue upper bound

    // Set the YCrCb color ranges for red based on the slider data
    private final Scalar lowerRed = new Scalar(19.8, 164.0, 68.0);   // Adjusted red lower bound
    private final Scalar upperRed = new Scalar(171.4, 200.0, 120.0); // Adjusted red upper bound

    // Set the YCrCb color ranges for yellow based on the slider data
    private final Scalar lowerYellow = new Scalar(79.3, 140.0, 20.0);  // Adjusted yellow lower bound
    private final Scalar upperYellow = new Scalar(204.0, 225.0, 75.0); // Adjusted yellow upper bound

    @Override
    public Mat processFrame(Mat input) {
        // Convert the input image from RGB to YCrCb color space
        Mat ycrcb = new Mat();
        Imgproc.cvtColor(input, ycrcb, Imgproc.COLOR_RGB2YCrCb);

        // Generate masks for blue, red, and yellow colors based on YCrCb color ranges
        Mat maskBlue = new Mat();
        Mat maskRed = new Mat();
        Mat maskYellow = new Mat();
        
        // Step 1: Detect Blue and mask it out
        Core.inRange(ycrcb, lowerBlue, upperBlue, maskBlue);
        processColor(maskBlue, input, new Scalar(255, 0, 0), "Blue", 500);  // Detect Blue
        
        // Mask out blue region so it's not detected as red or yellow
        Core.bitwise_not(maskBlue, maskBlue);
        Core.bitwise_and(ycrcb, ycrcb, ycrcb, maskBlue);  // Mask out blue region

        // Step 2: Detect Red on the remaining image
        Core.inRange(ycrcb, lowerRed, upperRed, maskRed);
        processColor(maskRed, input, new Scalar(0, 255, 0), "Red", 500);   // Detect Red

        // Mask out red region so it's not detected as yellow
        Core.bitwise_not(maskRed, maskRed);
        Core.bitwise_and(ycrcb, ycrcb, ycrcb, maskRed);  // Mask out red region

        // Step 3: Detect Yellow on the remaining image
        Core.inRange(ycrcb, lowerYellow, upperYellow, maskYellow);
        processColor(maskYellow, input, new Scalar(0, 255, 255), "Yellow", 500);  // Detect Yellow

        // Release Mat objects to prevent memory leaks
        ycrcb.release();
        maskBlue.release();
        maskRed.release();
        maskYellow.release();

        return input;
    }

    private void processColor(Mat mask, Mat frame, Scalar boxColor, String colorName, int minArea) {
        // Find contours on the mask
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            // Get bounding box for the contour
            Rect rect = Imgproc.boundingRect(contour);

            // Filter out small contours by area
            if (Imgproc.contourArea(contour) > minArea) {
                // Draw the bounding box
                Imgproc.rectangle(frame, rect, boxColor, 2);

                // Put text label near the bounding box
                Imgproc.putText(frame, colorName, new Point(rect.x, rect.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, boxColor, 2);
            }
        }

        // Release resources
        hierarchy.release();
        for (MatOfPoint contour : contours) {
            contour.release();
        }
    }

    @Override
    public void onViewportTapped() {
        // This method will be executed when the image display is clicked/tapped
        // You can add code here if you need to handle a UI tap event
    }
}
