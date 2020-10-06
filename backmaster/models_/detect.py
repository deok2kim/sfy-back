import argparse
from io import BytesIO
from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *
import sys
import io
import datetime

def detect(opt, save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0'
    # or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') or source.endwith('.jpg')
    
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    
    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    ### 함수에 못들어갈 경우 지역변수가 설정되어 지역변수 에러 발생하므로 최초로 여기에 선언
    main_list = [] 
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, test_img, vid_cap in dataset:
        orgin_img = test_img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()
        
        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            # tensor값 출력 det
            #print(det)
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # 이미지 사이즈 출력
            #s += '%gx%g ' % img.shape[2:]  

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    ###############################################################################################
    # 인식 모델의 동작 결과를 표시하는 곳 입니다.
    # s라는 문자열에 동작 결과를 저장해 반복문 후 출력해 모든 정보를 표시합니다.
        
                # 전체 결과 저장
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # n = 이미지 내 해당 라벨의 총 갯수, names[int(c)]=인식된 객체의 라벨명
                    

                # Write results
                # 각 객체별 넘버링을 수행, 저장
                number=2
                print()
                main_list = []
                img_count = 1 
                for *xyxy, conf, cls in reversed(det):
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        
                    # 출력 파일 내 라벨링 조절
                    if save_img or view_img:  # Add bbox to image
                        name = names[int(cls)]
                        if name == 'teddy bear': name='dog'
                        # 인식 객체의 라벨 이름 지정
                        label = '%d) %s %.2f ' % (math.log(number, 2), name, conf)
                        
                        # 인식된 객체별로 각각 저장 
                        # 개별로 저장된 파일명 형태 : 원본이미지이름_인식객체라벨_총인식된객체중몇번째객체인지 
                        each_img_box=(int(xyxy[0].item()),int(xyxy[1].item()),int(xyxy[2].item()),int(xyxy[3].item()))
                        crop_img = orgin_img.crop(each_img_box)
                        crop_img_name="{}.jpg".format("{0:0d}".format(img_count))
                        
                        savename=save_path[:-4]+"_"+name+"_"+crop_img_name
                        crop_img.save(savename)
                        img_count += 1
                        # now = str(datetime.datetime.now())[:10]
                        # savemedia='C:/Users/multicampus/Desktop/s03p23c106/backmaster/static/'+now+"_"+name+"_"+crop_img_name
                        # crop_img.save('C:/Users/multicampus/Desktop/s03p23c106/backmaster/static/'+name+"_"+now+"_"+crop_img_name)
                        # print(savename)
                        # print(savemedia)
                        
                        #라벨명, 좌상단x, 좌상단y, 우하단x, 우하단y
                        temp_list = [name,int(xyxy[0].item()),int(xyxy[1].item()),int(xyxy[2].item()),int(xyxy[3].item())]
                        main_list.append(temp_list)
                        
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        number += number
                    
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # 이미지 하나 분석 종료
            print()
     ###############################################################################################
           
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        #print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    print(f'메인리스트 {main_list}')
    return main_list
    
def detect_start(image_name):
    import argparse 
    # image = image.read().decode('utf-8')
    # image = image.read()
    # with open(image, 'r', encoding='utf-16') as f:
    #     lines  = f.readlines()
    #     print(lines)

    # if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detect.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='models_/weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='media/'+ image_name, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # print(parser,'sssssss')
    # print(type(image))
    # print('위엫ㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎ')
    
    opt = parser.parse_args(args=[])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    print('함수 실행')


    with torch.no_grad():
        lst = detect(opt)
        return lst