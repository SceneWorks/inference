"""Bounded offline acceptance runner; native executable owns model execution.

Preparatory/reference Python is permitted for S1, never part of product routing.
This runner records failures and leaves the prototype gate INCOMPLETE until
all three clips, both controls, SeedVR2 and human visual rubric are present.
"""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil

RECIPE="36cb612ec3df30094eeb4fec66f1528925ebba73"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def command(args, **kwargs):
    subprocess.run([str(a) for a in args],check=True,**kwargs)


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-show_format","-of","json",str(path)]))


def mux(rgb, source, out):
    # Complete raw picture is the duration owner. No -shortest: early audio
    # must never discard frames. Copy the normalized fixture's AAC soundtrack.
    command(["ffmpeg","-v","error","-y","-f","rawvideo","-pixel_format","rgb24","-video_size","1024x576","-framerate","24","-i",rgb,"-i",source,
             "-map","0:v:0","-map","1:a?","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-c:a","copy",out])
    video=next(s for s in probe(out)["streams"] if s["codec_type"]=="video")
    if (video["width"],video["height"],int(video["nb_read_frames"]))!=(1024,576,39):
        raise ValueError("native output geometry/frame count mismatch")
    timestamps=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_frames","-show_entries","frame=best_effort_timestamp_time","-of","json",str(out)]))["frames"]
    tick=1/12288
    if any(abs(float(frame["best_effort_timestamp_time"])-index/24)>tick for index,frame in enumerate(timestamps)):
        raise ValueError("fixed-case output frame PTS mismatch")
    source_audio=[s for s in probe(source)["streams"] if s["codec_type"]=="audio"]
    output_audio=[s for s in probe(out)["streams"] if s["codec_type"]=="audio"]
    if len(source_audio)!=len(output_audio):
        raise ValueError("source soundtrack stream count changed")
    for before,after in zip(source_audio,output_audio):
        audio_tick=1/int(before["sample_rate"])
        if any(abs(float(before.get(key,0))-float(after.get(key,0)))>audio_tick for key in ("start_time","duration")):
            raise ValueError("source soundtrack timing changed")


def native(args, source_rgb, out_rgb, denoise, guided):
    command_args=[args.binary,"--rgb24",source_rgb,"--root",args.root,"--upscaler",args.upscaler,"--lora",args.lora,
                  "--out",out_rgb,"--denoise",str(denoise),"--guide",str(guided).lower(),"--noise-fixture",args.noise]
    return execute(args,command_args,out_rgb)


def execute(args, command_args, out_rgb):
    import os
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=args.gpu_uuid
    peak={"host_rss_bytes":0,"device_used_mib":0,"samples":0}
    stopped=threading.Event()
    with out_rgb.with_suffix(".log").open("w",encoding="utf-8") as log:
        started=time.monotonic()
        process=subprocess.Popen([str(v) for v in command_args],stdout=log,stderr=subprocess.STDOUT,env=env)
        def sample():
            while not stopped.is_set():
                try:
                    peak["host_rss_bytes"]=max(peak["host_rss_bytes"],psutil.Process(process.pid).memory_info().rss)
                    used=subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
                    peak["device_used_mib"]=max(peak["device_used_mib"],int(used.strip()))
                    peak["samples"]+=1
                except (OSError,ValueError,psutil.Error,subprocess.CalledProcessError):
                    pass
                stopped.wait(.2)
        watcher=threading.Thread(target=sample,daemon=True); watcher.start()
        code=process.wait(); stopped.set(); watcher.join()
    return {"exit_code":code,"wall_seconds":time.monotonic()-started,"peak":peak,
            "executable_sha256":sha(Path(command_args[0])),
            "memory_method":"200ms native process RSS and assigned GPU device-used samples; measured sampled peaks, not allocation exact maxima",
            "log":out_rgb.with_suffix(".log").name,
            "stages":json.loads(out_rgb.with_suffix(".stages.json").read_text()) if out_rgb.with_suffix(".stages.json").exists() else None}


def main():
    parser=argparse.ArgumentParser()
    for name in ("binary","root","upscaler","lora","noise","manifest","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--cases",nargs="+",default=["h3","other-model","live-action"])
    parser.add_argument("--variants",nargs="+",default=["guided","unguided","latent-only"])
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"report.json").exists():
        raise SystemExit("use a fresh evidence directory; preserve prior measurements and mapped reference inputs")
    provenance=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    for component,path in zip(provenance["assets"],[args.lora,args.upscaler]):
        if sha(path)!=component["hash"]:
            raise SystemExit("corrupt or unsupported installed asset: "+str(path))
    report={"gate":"INCOMPLETE","decision":None,"upstream_commit":RECIPE,
            "hardware":subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=name,uuid,memory.total,driver_version","--format=csv,noheader"],text=True).strip(),
            "components":provenance["assets"],"cases":[],"noise_fixture_sha256":sha(args.noise),
            "required_visual_rubric":"useful detail on2/3 including nonH3; no material identity/motion/timing regression; numerical fidelity; measured feasible path; complementary SeedVR2 comparison",
            "production":"unstarted; no advertised capability","missing":["completed3case readout","SeedVR2 side-by-side","visual quality/risk rubric"]}
    report_path=args.out/"report.json"
    for kind in args.cases:
        entry=next(s for s in provenance["sources"] if s["kind"]==kind)
        source=Path(entry["path"])
        case_dir=args.out/kind; case_dir.mkdir(exist_ok=True)
        fixture=case_dir/"source.mp4"; source_rgb=case_dir/"source.rgb"
        offset={"h3":0,"other-model":59.1,"live-action":1}[kind]
        command(["ffmpeg","-v","error","-y","-ss",str(offset),"-i",source,"-t","1.625","-vf","fps=24,scale=512:288:flags=lanczos","-frames:v","39","-c:v","libx264","-crf","0","-pix_fmt","yuv420p","-c:a","aac",fixture])
        command(["ffmpeg","-v","error","-y","-i",fixture,"-map","0:v:0","-f","rawvideo","-pix_fmt","rgb24",source_rgb])
        bicubic=case_dir/"bicubic.mp4"
        command(["ffmpeg","-v","error","-y","-i",fixture,"-vf","scale=1024:576:flags=bicubic","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-c:a","copy",bicubic])
        case={"kind":kind,"source_provenance":{k:v for k,v in entry.items() if k!="path"},"original_sha256":sha(source),
              "normalization":{"start_seconds":offset,"fps":24,"width":512,"height":288,"frames":39,"resize":"lanczos","audio":"source AAC transcode; normalized test fixture only"},
              "source_fixture_sha256":sha(fixture),"bicubic_sha256":sha(bicubic),"source_probe":probe(fixture),"variants":[]}
        report["cases"].append(case)
        for variant in args.variants:
            out_rgb=case_dir/(variant+".rgb")
            outcome=native(args,source_rgb,out_rgb,0. if variant=="latent-only" else .1,variant=="guided")
            outcome["variant"]=variant; case["variants"].append(outcome)
            if outcome["exit_code"]==0:
                result=case_dir/(variant+".mp4"); mux(out_rgb,fixture,result)
                outcome.update({"result_sha256":sha(result),"result_probe":probe(result),"result":str(result)})
            else:
                report["runtime_failure"]={"case":kind,"variant":variant,"log":str(out_rgb.with_suffix('.log'))}
            report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
            if outcome["exit_code"]:
                raise SystemExit("native execution failed; preserved truthful incomplete report")
    report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
