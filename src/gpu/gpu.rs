use cust::memory::{DeviceBuffer, CopyDestination};
use cust::error::CudaResult;
use cust::memory::DeviceCopy;

pub struct GpuContext;

impl GpuContext {
    pub fn copy_to_device<T>(data: &[T]) -> CudaResult<DeviceBuffer<T>>
    where
        T: DeviceCopy,
    {
        DeviceBuffer::from_slice(data)
    }

    pub fn copy_to_host<T>(device_data: &DeviceBuffer<T>, host_data: &mut [T]) -> CudaResult<()>
    where
        T: DeviceCopy,
    {
        device_data.copy_to(host_data)
    }
}
