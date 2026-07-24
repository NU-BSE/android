#!/usr/bin/env python3
"""Collect screenshot/tree/window records around AccessibilityService events.

The service emits JSON to logcat tag DRLQN_TRACE. This host collector preserves
all timestamps; training must align modalities by timestamp, never file order.
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from pathlib import Path


def adb(serial,*args,binary=False):
    cmd=["adb"]+(["-s",serial] if serial else [])+list(args)
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout if binary else result.stdout.decode(errors="replace")


def component(serial):
    text=adb(serial,"shell","dumpsys","window","windows")
    match=re.search(r"mCurrentFocus=.*? ([A-Za-z0-9_.]+)/(\S+)}?",text)
    return (match.group(1),match.group(2).rstrip("}")) if match else ("","")


def viewport(serial):
    size=adb(serial,"shell","wm","size"); density=adb(serial,"shell","wm","density")
    sm=re.findall(r"(?:Override|Physical) size: (\d+)x(\d+)",size)
    dm=re.findall(r"(?:Override|Physical) density: (\d+)",density)
    width,height=map(int,sm[-1]) if sm else (0,0)
    return {"width_px":width,"height_px":height,"density":int(dm[-1])/160 if dm else 1.0}


def events(serial):
    cmd=["adb"]+(["-s",serial] if serial else [])+["logcat","-v","epoch","-s","DRLQN_TRACE:I","*:S"]
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True,bufsize=1)
    assert process.stdout
    for line in process.stdout:
        start=line.find("{")
        if start>=0:
            try: yield json.loads(line[start:])
            except json.JSONDecodeError: continue


def collect(output,episode_id,serial):
    output.mkdir(parents=True,exist_ok=True)
    (output/"screenshots").mkdir(exist_ok=True); (output/"trees").mkdir(exist_ok=True)
    with (output/"trajectory.jsonl").open("a",encoding="utf-8") as target:
        for sequence,event in enumerate(events(serial)):
            stem=f"{episode_id}-{sequence:06d}"
            screenshot_ts=time.time_ns()//1_000_000
            (output/"screenshots"/f"{stem}.png").write_bytes(adb(serial,"exec-out","screencap","-p",binary=True))
            remote="/sdcard/drlqn_window.xml"
            tree_ts=time.time_ns()//1_000_000
            adb(serial,"shell","uiautomator","dump","--compressed",remote)
            xml=adb(serial,"exec-out","cat",remote)
            (output/"trees"/f"{stem}.xml").write_text(xml,encoding="utf-8")
            package,activity=component(serial)
            record={"episode_id":episode_id,"sequence":sequence,"event":event,
                "event_timestamp_ms":int(event.get("timestamp_ms",screenshot_ts)),
                "screenshot_timestamp_ms":screenshot_ts,"accessibility_timestamp_ms":tree_ts,
                "screenshot_path":f"screenshots/{stem}.png","accessibility_path":f"trees/{stem}.xml",
                "package":package,"activity":activity,"viewport":viewport(serial)}
            target.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n"); target.flush()
            print(json.dumps({"collected":sequence,"package":package,"activity":activity}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--episode-id",default=time.strftime("%Y%m%d-%H%M%S")); parser.add_argument("--serial")
    args=parser.parse_args(); collect(args.output,args.episode_id,args.serial)


if __name__=="__main__": main()
