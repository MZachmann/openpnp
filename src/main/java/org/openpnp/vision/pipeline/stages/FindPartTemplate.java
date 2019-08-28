package org.openpnp.vision.pipeline.stages;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openpnp.spi.Nozzle;
import org.openpnp.util.OpenCvUtils;
import org.openpnp.vision.pipeline.CvPipeline;
import org.openpnp.vision.pipeline.CvStage;
import org.openpnp.vision.pipeline.CvStage.Result;
import org.openpnp.vision.pipeline.CvStage.Result.TemplateMatch;
import org.openpnp.vision.pipeline.Property;
import org.openpnp.vision.pipeline.Stage;
import org.pmw.tinylog.Logger;
import org.simpleframework.xml.Attribute;

/**
 * Tries to find a template image in the current image. This uses OpenCv pattern
 * match. It first rotates the image through 360 in large steps finding a
 * starting point angle. Then it does a binary search of rotation to find the
 * exact angle. It returns a single rotated rect the size of the template image.
 */
@Stage(category = "Image Processing", description = "Searches through the current image for a rectangular part. Returns a rotated rect.")

public class FindPartTemplate extends CvStage {
	@Attribute(required = false)
	@Property(description = "Enable logging.")
	private boolean log = false;

	public boolean isLog() {
		return log;
	}

	public void setLog(boolean log) {
		this.log = log;
	}

	@Attribute
	@Property(description = "Name of a prior stage to load the template image from.")
	private String templateStageName;

	@Attribute(required = false)
	@Property(description = "If score is below this value, then no matches will be reported. Default is 0.3.")
	private double threshold = 0.3;

	@Attribute(required = false)
	@Property(description = "Step amount to rotate for first recognition phase. Default is 22.5 degrees.")
	private double rotateStep = 22.5;

	@Attribute(required = false)
	@Property(description = "Angle resolution required. Default is 1.0 degree.")
	private double angleResolution = 1.0;

	/**
	 * Normalized recognition threshold in the interval [0,1]. Used to determine
	 * best match of candidates. For CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED, CV_TM_CCORR,
	 * and CV_TM_CCORR_NORMED methods, this is a minimum threshold for positive
	 * recognition; for all other methods, it is a maximum threshold. Default is
	 * 0.85.
	 */
	@Attribute(required = false)
	@Property(description = "Normalized minimum recognition threshold for the CCOEFF_NORMED method, in the interval [0,1]. Default is 0.85.")
	private double corr = 0.85f;

	public String getTemplateStageName() {
		return templateStageName;
	}

	public void setTemplateStageName(String templateStageName) {
		this.templateStageName = templateStageName;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public double getCorr() {
		return corr;
	}

	public void setCorr(double corr) {
		this.corr = corr;
	}

	public double getRotateStep() {
		return rotateStep;
	}

	public void setRotateStep(double rot) {
		this.rotateStep = rot;
	}

	public double getAngleResolution() {
		return angleResolution;
	}

	public void setAngleResolution(double res) {
		this.angleResolution = res;
	}

	private Rect clipRectangle;
	private Mat clippedImg;
	private Mat templateImg;

	// this is (almost) the MatchTemplate code
	// this finds the spot in the image at which the template best matches the image
	private List<TemplateMatch> findMatches(Mat template, Mat img) {
		Mat result = new Mat();

		Imgproc.matchTemplate(img, template, result, Imgproc.TM_CCOEFF_NORMED);

		MinMaxLocResult mmr = Core.minMaxLoc(result);
		double maxVal = mmr.maxVal;

		double rangeMin = Math.max(threshold, corr * maxVal);
		double rangeMax = maxVal;

		List<TemplateMatch> matches = new ArrayList<>();
		for (java.awt.Point point : OpenCvUtils.matMaxima(result, rangeMin, rangeMax)) {
			int x = point.x;
			int y = point.y;
			TemplateMatch match = new TemplateMatch(x, y, template.cols(), template.rows(), result.get(y, x)[0]);
			matches.add(match);
		}

		Collections.sort(matches, new Comparator<TemplateMatch>() {
			@Override
			public int compare(TemplateMatch o1, TemplateMatch o2) {
				return ((Double) o2.score).compareTo(o1.score);
			}
		});

		return matches;
	}

	// rotate the searched image and try to find the template
	// we don't rotate the template because of size and border
	private TemplateMatch findRotatedMatch(double angle) {

		// rotate the src image
		Point center = new Point(clipRectangle.width / 2.0, clipRectangle.height / 2.0);
		Mat mapMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);
		Mat dst = new Mat(clipRectangle.width, clipRectangle.height, clippedImg.type());
		Imgproc.warpAffine(clippedImg, dst, mapMatrix, clipRectangle.size(), Imgproc.INTER_LINEAR);

		// run a template match
		List<TemplateMatch> matches = findMatches(templateImg, dst);
		TemplateMatch tm = new TemplateMatch(0, 0, 0, 0, 0);

		// if we had matches, find the best
		if (!matches.isEmpty()) {
			// they are sorted so this is the best match
			tm = matches.get(0);

			// unrotate the center point.
			// tm.x,tm.y are the upper left corner of the match rectangle
			double theta = Math.PI * angle / 180.0;
			double px = tm.x + templateImg.cols() / 2.0 - center.x;
			double py = tm.y + templateImg.rows() / 2.0 - center.y;
			tm.x = center.x + (px * Math.cos(theta) - py * Math.sin(theta));
			tm.y = center.y + (px * Math.sin(theta) + py * Math.cos(theta));
		}

		// and return it
		return tm;
	}

	@Override
	public Result process(CvPipeline pipeline) throws Exception {
		if (templateStageName == null) {
			return null;
		}

		// read the template image
		Mat workImg = pipeline.getWorkingImage();
		Result template = pipeline.getResult(templateStageName);
		if (template == null) {
			return null;
		}
		templateImg = template.image;

		// get the size of the template and clip the working image
		Point ptTemplateSize = new Point(template.image.cols(), template.image.rows());

		// allow the part pick to be pretty far away from center
		// this rectangle doesn't need to be exactly centered
		int sx = (int) (2.5 * Math.max(ptTemplateSize.x, ptTemplateSize.y)); // 2.5 times template max dimension?
		sx = Math.min(Math.min(sx, workImg.cols()), workImg.rows()); // the template takes up half the image?
		if (0 != (sx & 1)) {
			sx--; // ensure that the size is odd so there's a center point
		}
		int topleftx = (workImg.cols() - sx) / 2;
		int toplefty = (workImg.rows() - sx) / 2;
		clipRectangle = new Rect(topleftx, toplefty, sx, sx);

		// search in a square chunk at the center of the image
		clippedImg = workImg.submat(clipRectangle);

		double angle = -rotateStep; // start at 0
		TemplateMatch bestMatch = new TemplateMatch(0.0, 0.0, 0.0, 0.0, 0.0);
		double bestAngle = 0;

		// check at the expected angle first
		Nozzle noz = (Nozzle) pipeline.getProperty("nozzle");
		if(noz == null) {
			angle = 0;
		}
		else
		{
			angle = noz.getLocation().getRotation();
		}

		// test at the expected angle first
		{
			TemplateMatch best = findRotatedMatch(angle); // since they're sorted
			if(rotateStep > angleResolution) {
				if(best.score == 0)
				{
					// perturb if not found
					angle -= rotateStep;
					best = findRotatedMatch(angle); // since they're sorted
				}
				if(best.score == 0)
				{
					// perturb if not found
					angle += 2*rotateStep;
					best = findRotatedMatch(angle); // since they're sorted
				}
			}
			bestMatch = best;
			bestAngle = angle;
		}

		if (bestMatch.score > 0) {
			Logger.debug("Best match, first phase= {}", bestMatch);
		} else {
			Logger.debug("No template match found");
		}

		// now do a binary search from the best angle
		if (bestMatch.score > 0) {
			double delta = rotateStep / 2;
			boolean direction = false;
			while (delta > angleResolution) {
				angle = bestAngle + delta * (direction ? -1 : 1);
				TemplateMatch best = findRotatedMatch(angle); // since they're sorted
				if (best.score > bestMatch.score) {
					bestMatch = best;
					bestAngle = angle;
				} else {
					if (direction) {
						delta = delta / 2;
					}
					direction = !direction;
				}
			}
			Logger.debug("Best match, second phase= {}", bestMatch);
		} else {
			// we didn't find a part, return the center rectangle?
			bestMatch.x = clipRectangle.width / 2;
			bestMatch.y = clipRectangle.height / 2;
			bestAngle = 0;
		}

		// create the rotated rect drawing the outline of the part template
		// X and Y dimensions looking up are backwards from the top camera
		Point pt = new Point(bestMatch.x + clipRectangle.x, bestMatch.y + clipRectangle.y);
		Size sz = new Size(ptTemplateSize.x, ptTemplateSize.y);
		RotatedRect rcout = new RotatedRect(pt, sz, bestAngle); // gets angle wrong
		Logger.debug("Rotated rectangle from match = {} with center {}x{}", rcout, rcout.center.x, rcout.center.y);

		// let's return null if we didn't find a match
		return new Result(workImg, (bestMatch.score > 0) ? rcout : null);
	}

}
