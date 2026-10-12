// Primitive composition adapted from shadcn's Base Nova registry; original project styling.
import { Select } from "@base-ui/react/select";
import { Tooltip } from "@base-ui/react/tooltip";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { Toggle } from "@base-ui/react/toggle";
import { Check, ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

export function Choice<T extends string>({
  label,
  value,
  onChange,
  items,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  items: { value: T; label: string }[];
}) {
  return (
    <div className="choice">
      <span className="control-label">{label}</span>
      <Select.Root
        items={items}
        value={value}
        onValueChange={(v) => v && onChange(v as T)}
      >
        <Select.Trigger className="select-trigger" aria-label={label}>
          <Select.Value />
          <Select.Icon>
            <ChevronDown size={15} />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Positioner
            sideOffset={8}
            align="start"
            alignItemWithTrigger={false}
            className="select-positioner"
          >
            <Select.Popup className="select-popup">
              <Select.ScrollUpArrow className="select-scroll">
                ↑
              </Select.ScrollUpArrow>
              <Select.List>
                {items.map((item) => (
                  <Select.Item
                    className="select-item"
                    key={item.value}
                    value={item.value}
                  >
                    <Select.ItemText>{item.label}</Select.ItemText>
                    <Select.ItemIndicator>
                      <Check size={15} />
                    </Select.ItemIndicator>
                  </Select.Item>
                ))}
              </Select.List>
              <Select.ScrollDownArrow className="select-scroll">
                ↓
              </Select.ScrollDownArrow>
            </Select.Popup>
          </Select.Positioner>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}
export function Hint({
  children,
  text,
}: {
  children: ReactNode;
  text: ReactNode;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger className="hint-trigger">{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Positioner sideOffset={8}>
          <Tooltip.Popup className="tooltip">{text}</Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
export function Segments({
  label,
  value,
  onChange,
  items,
  className = "compact-tabs",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  items: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <ToggleGroup
      className={className}
      aria-label={label}
      value={[value]}
      onValueChange={(v) => {
        if (v.length) onChange(v[0]);
      }}
    >
      {items.map((item) => (
        <Toggle key={item.value} value={item.value}>
          {item.label}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}
