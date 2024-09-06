#!/usr/bin/env wolframscript
(* ::Package:: *)

spinmax = 10000;
(*Extrapolate the SU2-vertex amplitude*)
(*QVamp= Import["/home/ri47hud/codes/Frusta quantum amplitude/Cluster Files/Q_amps/SU2_Vamp_full.csv"];
SemiVamp = Import["/home/ri47hud/codes/Frusta quantum amplitude/Cluster Files/Q_amps/SU2_Vamp_semi_large.csv"];
jmodel[x_,a_,b_,c_]:=a/(Exp[b*x]-c);
ks=Symbol["k"<>ToString[#]]& /@Range[7];
as=Symbol["a"<>ToString[#]]& /@Range[7];
bs=Symbol["b"<>ToString[#]]& /@Range[7];
cs=Symbol["c"<>ToString[#]]& /@Range[7];
fits1 = Array[Function[i,FindFit[((SemiVamp[[i,1;;8]]-QVamp[[i,1;;8]])/SemiVamp[[i,1;;8]]),jmodel[ks[[i]],as[[i]],bs[[i]],cs[[i]]],{{as[[i]],-0.8},{bs[[i]],-0.2},{cs[[i]],1.0}},ks[[i]]]],3,1];
fits2 = Array[Function[i,FindFit[((SemiVamp[[i,1;;7]]-QVamp[[i,1;;7]])/SemiVamp[[i,1;;7]]),jmodel[ks[[i]],as[[i]],bs[[i]],cs[[i]]],{{as[[i]],-0.8},{bs[[i]],-0.2},{cs[[i]],1.0}},ks[[i]]]],3,4];
fits3 = Array[Function[i,FindFit[((SemiVamp[[i,1;;6]]-QVamp[[i,1;;6]])/SemiVamp[[i,1;;6]]),jmodel[ks[[i]],as[[i]],bs[[i]],cs[[i]]],{{as[[i]],-0.8},{bs[[i]],-0.2},{cs[[i]],1.0}},ks[[i]]]],1,7];
fits = Join[fits1,fits2,fits3];
relerrVampK = Table[jmodel[k,as[[j]],bs[[j]],cs[[j]]]/.fits[[j]],{j,7},{k,spinmax}];
ExtraVampK = Table[SemiVamp[[j,k]]*(1-relerrVampK[[j,k]]),{j,7},{k,spinmax}];
kmodel[x_,a_,b_,c_]:=a/(Exp[b*x]-c);
js=Symbol["j"<>ToString[#]]& /@Range[spinmax];
ds=Symbol["d"<>ToString[#]]& /@Range[spinmax];
es=Symbol["e"<>ToString[#]]& /@Range[spinmax];
fs=Symbol["f"<>ToString[#]]& /@Range[spinmax];
jfits = Array[FindFit[(SemiVamp[[1;;7,#]]-ExtraVampK[[1;;7,#]])/SemiVamp[[1;;7,#]],kmodel[js[[#]],ds[[#]],es[[#]],fs[[#]]],{{ds[[#]],-0.3},{es[[#]],-0.1},{fs[[#]],1.5}},js[[#]]]&,spinmax,1];
relerrVamp = Table[kmodel[j,ds[[k]],es[[k]],fs[[k]]]/.jfits[[k]],{j,spinmax},{k,spinmax}];
ExtraVamp = Table[SemiVamp[[j,k]]*(1-relerrVamp[[j,k]]),{j,spinmax},{k,spinmax}];
Export["/home/ri47hud/codes/Frusta quantum amplitude/Cluster Files/SU2_Vamp_extra.csv", ExtraVamp, "CSV"]*)

(*Extrapolate the SU2-edge amplitude*)
QEamp= Import["/ssd/ri47hud/codes/Frusta quantum amplitude/Cluster Files/SU2_Eamp_full.csv"];
SemiEamp = Import["/ssd/ri47hud/codes/Frusta quantum amplitude/Cluster Files/SU2_Eamp_semi.csv"];
jmodel[x_,a_,b_,c_]:=a*Exp[-b*x]+c;
js=Symbol["j"<>ToString[#]]& /@Range[spinmax];
as=Symbol["a"<>ToString[#]]& /@Range[spinmax];
bs=Symbol["b"<>ToString[#]]& /@Range[spinmax];
cs=Symbol["c"<>ToString[#]]& /@Range[spinmax];
fits = Array[Function[i,FindFit[(Abs[SemiEamp[[1;;8,i]]-QEamp[[1;;8,i]]]/SemiEamp[[1;;8,i]]),jmodel[js[[i]],as[[i]],bs[[i]],cs[[i]]],{as[[i]],bs[[i]],cs[[i]]},js[[i]]]],8];
RelerrEampJ = Table[jmodel[j,as[[k]],bs[[k]],cs[[k]]]/.fits[[k]],{j,spinmax},{k,8}];
ExtraEampJ = Table[SemiEamp[[j,k]]*(1+RelerrEampJ[[j,k]]),{j,spinmax},{k,8}];
kmodel[x_,a_,b_,c_]:=a/(Exp[b*x]-c)
ks=Symbol["k"<>ToString[#]]& /@Range[spinmax];
ds=Symbol["d"<>ToString[#]]& /@Range[spinmax];
es=Symbol["e"<>ToString[#]]& /@Range[spinmax];
fs=Symbol["f"<>ToString[#]]& /@Range[spinmax];
kfits = Array[FindFit[(Abs[SemiEamp[[#,1;;8]]-ExtraEampJ[[#,1;;8]]])/SemiEamp[[#,1;;8]],kmodel[ks[[#]],ds[[#]],es[[#]],fs[[#]]],{{ds[[#]],0.01},{es[[#]],0.06},{fs[[#]],0.9}},ks[[#]]]&,spinmax];
relerrEamp = Table[kmodel[k,ds[[j]],es[[j]],fs[[j]]]/.kfits[[j]],{j,spinmax},{k,spinmax}];
ExtraEamp = Table[SemiEamp[[j,k]]*(1+relerrEamp[[j,k]]),{j,spinmax},{k,spinmax}];
Export["/ssd/ri47hud/codes/Frusta quantum amplitude/Cluster Files/SU2_Eamp_extra.csv", ExtraEamp, "CSV"]
