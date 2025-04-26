// 354. Russian Doll Envelopes 变体版
// addNum O(log N)
// findMedian O(1)

class MedianFinder {
    private PriorityQueue<Integer> left;
    private PriorityQueue<Integer> right;

    public MedianFinder() {
        left = new PriorityQueue<Integer>((a, b) -> b - a);
        right = new PriorityQueue<Integer>();
    }
    
    public void addNum(int num) {
        left.offer(num);
        right.offer(left.poll());
        if (left.size() < right.size()) {
            left.offer(right.poll());
        }
    }
    
    public double findMedian() {
        return (left.size() + right.size()) % 2 == 1 ? 
            left.peek() : (left.peek() + right.peek()) / 2.0;
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */