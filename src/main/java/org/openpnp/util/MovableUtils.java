package org.openpnp.util;

import java.util.HashMap;
import java.util.Map;

import org.openpnp.model.Configuration;
import org.openpnp.model.Location;
import org.openpnp.spi.Head;
import org.openpnp.spi.HeadMountable;
import org.pmw.tinylog.Logger;

public class MovableUtils {
    /**
     * Moves the given HeadMountable to the specified Location by first commanding the head to
     * safe-Z all of it's components, then moving the HeadMountable in X, Y and C, followed by
     * moving in Z.
     * 
     * @param hm
     * @param location
     * @param speed
     * @throws Exception
     */
    public static void moveToLocationAtSafeZ(HeadMountable hm, Location location, double speed)
            throws Exception {
        Head head = hm.getHead();
        head.moveToSafeZ(speed);
        hm.moveTo(location.derive(null, null, Double.NaN, null), speed);
        hm.moveTo(location, speed);
    }

    public static void moveToLocationAtSafeZ(HeadMountable hm, Location location) throws Exception {
        moveToLocationAtSafeZ(hm, location, hm.getHead().getMachine().getSpeed());
    }
    
    public static void park(Head head) throws Exception {
        head.moveToSafeZ();
        HeadMountable hm = head.getDefaultCamera();
        Location location = head.getParkLocation();
        location = location.derive(null, null, Double.NaN, Double.NaN);
        hm.moveTo(location);
        try {
            Map<String, Object> globals = new HashMap<>();
            globals.put("head", head);
            Configuration.get().getScripting().on("Head.AfterPark", globals);
        }
        catch (Exception e) {
            Logger.warn(e);
        }
    }
}
