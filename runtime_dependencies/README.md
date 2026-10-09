# Runtime files for the RK3588 follow car

This branch includes the files used by the default `run_request_0428_modular.sh`
configuration:

- `../models/yolo11n_int8_person_val2017.rknn` and
  `../models/osnet_x0_25_msmt17_b1.rknn`: the two active RKNN models.
- `../librknnrt.so`: the RK3588 RKNN runtime library used by the launcher.
- `arm64_openni2/`: the Astra OpenNI runtime and its Orbbec driver, copied from
  `/home/topeet/AstraSDK/arm64_openni2`. See `ASTRA_LICENSE.txt`.
- `lianzhan/src/lz30ema_rs485/`: the LZ30EMA RS485 Python driver copied from
  `/home/topeet/lianzhan/src/lz30ema_rs485`. The package metadata declares MIT.

The default INI points to the copies of OpenNI and the motor driver in this
directory. Run the launcher from the repository as documented in the main
README; it changes to the repository root before loading the INI. The
`ASTRA_DEPTH_OPENNI_PATH` and `MOTOR_RS485_LIB_DIR` environment variables can
override the two configured paths when needed.

The following board-side components are still installed separately: Python
3.10 packages `rknn-toolkit-lite2` 2.3.2, `openni` 2.3.0, NumPy, OpenCV,
SciPy and Requests; the RK3588 NPU kernel driver; camera and sensor drivers;
and the physical serial and IIO devices. The packaged ELF libraries are for
the current ARM64 RK3588 board, not other processor architectures. This
repository does not contain recordings, machine-specific device state or
installed Python site-packages.
