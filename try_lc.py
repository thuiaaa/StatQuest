class Solution():
    def twoSum(self, nums, target):
        num_dict = {}
        for index, num in enumerate(nums):
            complement = target - num
            if complement in num_dict:
                return [num_dict[complement], index]
            num_dict[num] = index
        return []


if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 13
    solver = Solution()
    result = solver.twoSum(nums, target)
    print(result)
