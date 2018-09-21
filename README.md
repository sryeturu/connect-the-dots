# connect-the-dots

Early stages of an automatic connect the dots solver through object detection.


# Example

![screenshot from 2018-09-21 18-04-26](https://user-images.githubusercontent.com/32404036/45909672-d5084d00-bdc8-11e8-8eb7-660a064d5131.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![screenshot from 2018-09-21 17-57-36](https://user-images.githubusercontent.com/32404036/45909523-d6854580-bdc7-11e8-8673-91a1056c4698.png)

```
python run.py --img test.png --cfg connect_the_dots.cfg --weights 600_iters.weights --out result
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![screenshot from 2018-09-21 17-57-36](https://user-images.githubusercontent.com/32404036/45909523-d6854580-bdc7-11e8-8673-91a1056c4698.png)

![screenshot from 2018-09-21 18-04-39](https://user-images.githubusercontent.com/32404036/45909673-d6397a00-bdc8-11e8-9c90-ab59db3fe314.png)


## TODO list:
- [x] Implement YoloV3 in tensorflow
- [x] Create training sample generator
- [x] Basic working model (very basic)
- [ ] Get better data (train on mnist? take more pics?)
- [ ] Generate better traning samples
- [ ] Train model again (rip wallet)
- [ ] ?
- [ ] ?
- [ ] ?
- [ ] Clean code
