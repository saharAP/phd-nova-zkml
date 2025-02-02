// measure_memory.rs

use std::cell::RefCell;
use std::rc::Rc;
use std::thread;
use std::time::Duration;

use jemalloc_ctl::{epoch, stats};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub struct MemoryTracker {
    baseline_allocated: Option<u64>,
    baseline_resident: Option<u64>,
    peak_allocated: u64, // NEW: Store peak allocated memory
    peak_resident: u64,  // NEW: Store peak resident memory
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            baseline_allocated: None,
            baseline_resident: None,
            peak_allocated: 0, // Initialize peak values
            peak_resident: 0,
        }
    }

    pub fn start_measurement(&mut self) {
        epoch::advance().unwrap();
        self.baseline_allocated = Some(stats::allocated::read().unwrap() as u64);
        self.baseline_resident = Some(stats::resident::read().unwrap() as u64);
        self.peak_allocated = self.baseline_allocated.unwrap_or(0); // Start peak tracking
        self.peak_resident = self.baseline_resident.unwrap_or(0);
    }

    pub fn track_peak_usage(&mut self) {
        epoch::advance().unwrap();
        let current_allocated = stats::allocated::read().unwrap() as u64;
        let current_resident = stats::resident::read().unwrap() as u64;

        if current_allocated > self.peak_allocated {
            self.peak_allocated = current_allocated;
        }
        if current_resident > self.peak_resident {
            self.peak_resident = current_resident;
        }
    }

    pub fn end_measurement(&self) -> (Option<f64>, Option<f64>, f64, f64) {
        epoch::advance().unwrap();
        let current_allocated = stats::allocated::read().unwrap() as u64;
        let current_resident = stats::resident::read().unwrap() as u64;
    
        let bytes_to_mb = |bytes: u64| bytes as f64 / 1_048_576.0; // Convert bytes to MB
    
        (
            self.baseline_allocated.map(|base| bytes_to_mb(current_allocated.saturating_sub(base))),
            self.baseline_resident.map(|base| bytes_to_mb(current_resident.saturating_sub(base))),
            bytes_to_mb(self.peak_allocated), // Convert peak values to MB
            bytes_to_mb(self.peak_resident),
        )
    }
    
}


#[macro_export]
macro_rules! measure_memory {
    ($func:expr) => {{
        use crate::memory_utils::MemoryTracker;

        let mut tracker = MemoryTracker::new();
        tracker.start_measurement();
        
        let result = {
            let output = $func;

            // Simulate periodic tracking of memory usage during execution
            for _ in 0..5 {
                tracker.track_peak_usage();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            output
        };

        let (allocated_delta, resident_delta, peak_allocated, peak_resident) = tracker.end_measurement();

        println!("Memory usage:");
        println!("  Peak allocated:  {:.3} MB", peak_allocated);
        println!("🚀  Peak resident:   {:.3} MB", peak_resident);

        result
    }};
}


