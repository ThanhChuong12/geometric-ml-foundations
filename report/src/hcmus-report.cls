\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmus-report}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage{anyfontsize,amsmath,amssymb}
\RequirePackage[T5]{fontenc}

\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{booktabs}
\RequirePackage[center]{caption}

\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

% Change to '[english,vietnamese]' to use Vietnamese labels
\RequirePackage[english]{babel}

\RequirePackage{tikz}
\RequirePackage{titlesec}

% Custom commands
\newcommand{\Month}{
  \ifcase \the\month
  \or Tháng 1
  \or Tháng 2
  \or Tháng 3
  \or Tháng 4
  \or Tháng 5
  \or Tháng 6
  \or Tháng 7
  \or Tháng 8
  \or Tháng 9
  \or Tháng 10
  \or Tháng 11
  \or Tháng 12
  \fi
}

\newcommand{\@uniname}{TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN, ĐHQG-HCM}

\newcommand{\@deptname}{KHOA CÔNG NGHỆ THÔNG TIN}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}

\newcommand{\@coursename}{Course name}
\newcommand{\coursename}[1]{\renewcommand{\@coursename}{#1}}

\newcommand{\@reporttype}{Assignment Report}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}

\renewcommand{\@title}{Report title}

\newcommand{\@advisor}{Advisor}
\newcommand{\advisor}[1]{\renewcommand{\@advisor}{#1}}

\newcommand{\@stuname}{STUDENT:\ H (xxxxxxx)}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}

\newcommand{\@reportplace}{Tp. Hồ Chí Minh}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}

\newcommand{\@reporttime}{\Month năm \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

% ============================================================
% CUSTOM: Định nghĩa màu và format cho các cấp heading
% ============================================================
\definecolor{chapterblue}{RGB}{70,130,180}
\definecolor{lightblue}{RGB}{135,206,235}

% Format cho Section (Chương 1, 2, 3...)
\titleformat{\section}
  {\normalfont\huge\bfseries\color{chapterblue}}
  {\begin{tikzpicture}[remember picture, overlay]
    \node[fill=chapterblue, text=white, font=\Large\bfseries, 
          minimum width=2.0cm, minimum height=2.5cm, align=center] 
          at (0,0) {Chương\\\thesection};
   \end{tikzpicture}\hspace{2cm}}
  {0pt}
  {}
\titlespacing*{\section}{0pt}{3.5ex plus 1ex minus .2ex}{7ex plus .2ex}

% Format cho Subsection (1.1, 1.2...)
\titleformat{\subsection}
  {\normalfont\Large\bfseries\color{chapterblue}}
  {\tikz[baseline=(box.base)]\node[fill=chapterblue, text=white, font=\Large\bfseries,
              inner sep=8pt, minimum height=1cm] (box) {\thesubsection};\hspace{0.8em}}
  {0pt}
  {}
  [\vspace{-1.0em}\hspace*{1.8cm}{\color{chapterblue}\titlerule[0.8pt]}]

% Format cho Subsubsection (1.1.1, 1.1.2...)
\titleformat{\subsubsection}
  {\normalfont\large\bfseries\color{chapterblue}}
  {\thesubsubsection.}
  {0.5em}
  {}

% Configurations
%\graphicspath{{subdir1/}{subdir2/}{subdir3/}...{subdirn/}}

\geometry{
  paper=a4paper,
  left=3cm,
  right=2cm,
  vmargin=2cm,
  includeheadfoot=true,
  headheight=35pt
}

\onehalfspacing
\raggedbottom
\renewcommand{\arraystretch}{1.2}

% Set depth of numbering for sections and table of contents
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% Set depth of numbering for counters
\AtBeginDocument{\counterwithin{equation}{section}}
\AtBeginDocument{\counterwithin{table}{section}}
\AtBeginDocument{\counterwithin{figure}{section}}

\AtBeginDocument{\renewcommand*\bibname{References}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}

\pagestyle{fancy}
\fancyhf{}

% The drawing units are interpreted as multiples of \unitlength
% unless a unit is given
%\setlength{\unitlength}{1pt}

\if@twoside
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmus_logo.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmus_logo.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmus_logo.png}}
      \end{picture}
    \end{tabular}
  }
\fi

\fancyfoot[C]{\thepage}

% Custom pages and sections
\newcommand{\coverpage}{
  \pagestyle{empty}

  \vspace*{-3\baselineskip}
  \begin{center}
    \MakeUppercase{\@uniname}

    \MakeUppercase{\@deptname}
    \vfill
    \includegraphics[scale=0.15]{graphics/hcmus_logo.png}
    \vfill
    \bfseries
    \begin{tabular}{c}
      \Large\@coursename                                 \\
      {}                                                 \\
      \midrule                                           \\
      \Large\@reporttype                                 \\
      {}                                                 \\
      \Huge\parbox[c]{0.8\linewidth}{\centering \@title} \\
      {}                                                 \\
      \bottomrule
    \end{tabular}
    \vfill
    \mdseries
    \begin{tabular}{rll}
      Giảng viên hướng dẫn: \@advisor \\
      Sinh viên: \@stuname \\
    \end{tabular}
    \vfill
    \MakeUppercase{\@reportplace, \@reporttime}
  \end{center}

  \cleardoublepage
  \pagestyle{fancy}
}
