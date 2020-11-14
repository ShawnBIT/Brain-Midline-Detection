import numpy as np
import pydicom

def parse_dicom_window(dcm):
    window_center_tag = (0x0028, 0x1050)
    window_width_tag = (0x0028, 0x1051)
    wc, ww = None, None
    if window_width_tag in dcm:
        if type(dcm[window_width_tag].value) == pydicom.multival.MultiValue:
            ww = float(dcm[window_width_tag].value[0])
        else:
            ww = float(dcm[window_width_tag].value)
    if window_center_tag in dcm:
        if type(dcm[window_center_tag].value) == pydicom.multival.MultiValue:
            wc = float(dcm[window_center_tag].value[0])
        else:
            wc = float(dcm[window_center_tag].value)
    return wc, ww

class DICOM(object):
    def __init__(self, dcm_path):
        self.dcm_path = dcm_path
        self.dcm = pydicom.read_file(dcm_path)
        try:
            self.instance_no = int(self.dcm[(0x0020, 0x0013)].value)
            slope_tag = (0x0028, 0x1053)
            intercept_tag = (0x0028, 0x1052)
            self.slope = float(self.dcm[slope_tag].value) if slope_tag in self.dcm else 1.0
            self.intercept = float(self.dcm[intercept_tag].value) if intercept_tag in self.dcm else 0.0
            self.pid = self.dcm[(0x0010, 0x0020)].value
            self.study_id = self.dcm[(0x0020, 0x000d)].value
            self.series_id = self.dcm[(0x0020, 0x000e)].value
            self.pixel_spacing = (float(self.dcm.PixelSpacing[0]), float(self.dcm.PixelSpacing[1]))
            self.img_height = int(self.dcm[(0x0028, 0x0010)].value)
            self.img_width = int(self.dcm[(0x0028, 0x0011)].value)
            try:
                self.slice_thickness = float(self.dcm[(0x0018, 0x0050)].value)
            except:
                pass
            try:
                self.image_loc = map(float, self.dcm[(0x0020, 0x0032)].value)
            except:
                print '[Warning]', self.dcm_path
                pass
            try:
                self.slice_loc = float(self.dcm[(0x0020, 0x1041)].value)
            except:
                self.slice_loc = self.image_loc[2]
                pass
            try:
                self.series_number = int(self.dcm[(0x0020, 0x0011)].value)
            except:
                self.series_number = 0

            self.window_center, self.window_width = parse_dicom_window(self.dcm)
            self.is_valid_ = True
        except KeyError as e:
            self.is_valid_ = False
            print str(e)
        except AttributeError as e:
            self.is_valid_ = False
            print str(e)

    def is_valid(self):
        return self.is_valid_

    def get_image(self, window_center=None, window_width=None, dual_threshold=None):
        """
        Extract image from dicom
        """
        try:
            image_u16 = self.dcm.pixel_array
        except ValueError, e:
            print e
            return np.zeros((0, 0))
#         except NotImplementedError:
#             image_u16, flag = gdcm_to_numpy(self.dcm_path)
#             if not flag:
#                 return numpy.zeros((0, 0))

        img_u16 = np.array(image_u16, dtype='float32') * self.slope + self.intercept
        if window_center is None or window_width is None:
            window_center, window_width = self.window_center, self.window_width
        if window_center is None or window_width is None:
            assert False, 'dcm file %s does not contain window information, must assign one' % self.dcm_path
        ww, wc = window_width, window_center
        win_up = wc + ww / 2
        win_down = wc - ww / 2
        # add dual_threshold by zzhou
        if dual_threshold is not None:
            assert type(dual_threshold) is tuple or type(dual_threshold) is list, type(dual_threshold)
            assert len(dual_threshold) == 2, dual_threshold
            down_tsd = min(dual_threshold)
            up_tsd = max(dual_threshold)
            if down_tsd > win_down:
                win_down = down_tsd
            if up_tsd < win_up:
                win_up = up_tsd
        img_u16[img_u16 > win_up] = win_up
        img_u16[img_u16 < win_down] = win_down
        ww = win_up - win_down
        img_u16 -= win_down
        img_8u = np.array(img_u16 * 255.0 / ww, dtype='uint8')
        return img_8u
